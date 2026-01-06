import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from core.utils.config import load_config
from core.data.base_dataset import SimpleDataset
from core.data.data_generator import SyntheticGaussianMixture
from core.modeling.backbone.simple_embedding_net import EmbeddingNet
from core.losses.mpcbml_loss import MpcbmlLoss
from core.data.evaluation.retrieval_metric import compute_recall_at_k
from core.utils.prototype_tracker import PrototypeTracker
from core.solver.build_optimizer import build_optimizer
from core.utils.visualization_logger import VisualizationLogger, save_prototypes_visualization
from core.utils.prototype_initializer_factory import PrototypeInitializerFactory

from collections import defaultdict
import csv
import numbers


def _is_number(x) -> bool:
    return isinstance(x, numbers.Number)


def _history_append(history: dict, epoch_stats: dict, epoch_idx: int):
    """
    Append one epoch record to history. Ensures all numeric keys become series of equal length.
    """
    history.setdefault("epoch", []).append(int(epoch_idx))
    for k, v in epoch_stats.items():
        if _is_number(v):
            history.setdefault(k, []).append(float(v))


def _write_history_csv(history: dict, out_csv_path: Path):
    """
    Writes a rectangular CSV table; missing values become empty.
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(history.keys())
    n = max((len(history[k]) for k in keys), default=0)

    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            row = []
            for k in keys:
                row.append(history[k][i] if i < len(history[k]) else "")
            w.writerow(row)


def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"🚀 {text}")
    print("=" * 70 + "\n")


def compute_train_embeddings(model, train_loader, device):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            feats = model(x)  # [B, embed_dim]
            all_feats.append(feats.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def flatten_protos(protos: torch.Tensor) -> torch.Tensor:
    # protos can be [C, K, D] or [C*K, D]
    if protos.dim() == 3:
        return protos.view(-1, protos.shape[-1])
    return protos


def _make_snapshot_epochs(EPOCHS: int):
    """
    Returns a sorted unique list of 1-based epochs at which to snapshot.
    Guarantees includes epoch 1 and final epoch.
    """
    candidates = [1, max(1, EPOCHS // 4), max(1, EPOCHS // 2), max(1, (3 * EPOCHS) // 4), EPOCHS]
    return sorted(set(int(x) for x in candidates if 1 <= int(x) <= EPOCHS))


def main():
    # ====================================================================
    # ARGUMENT PARSING & CONFIGURATION
    # ====================================================================
    parser = argparse.ArgumentParser(description="MP-CBML Training with YAML Configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mpcbml_toy.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    print_banner("LOADING CONFIGURATION")

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = load_config(args.config)
    print(f"✓ Loaded config from: {args.config}\n")

    # ====================================================================
    # DEVICE SETUP
    # ====================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {DEVICE}\n")

    # Extract config values
    N_CLASSES = cfg.LOSSES.MPCBML_LOSS.N_CLASSES
    K_PROTOS = cfg.LOSSES.MPCBML_LOSS.PROTOTYPE_PER_CLASS
    EMBED_DIM = cfg.MODEL.HEAD.DIM
    BATCH_SIZE = cfg.DATA.TRAIN_BATCHSIZE
    EPOCHS = cfg.SOLVER.MAX_EPOCHS

    # ====================================================================
    # VISUALIZATION LOGGER SETUP
    # ====================================================================
    print_banner("INITIALIZING VISUALIZATION SYSTEM")

    output_dir = Path(cfg.OUTPUT.SAVE_DIR)  # IMPORTANT: Path for reliable file ops
    viz_logger = VisualizationLogger(str(output_dir))
    print("✓ Visualization logger initialized")
    print(f"✓ Output directory: {output_dir}\n")

    # ====================================================================
    # DATA PREPARATION
    # ====================================================================
    print_banner("DATA PREPARATION")

    print("Generating synthetic dataset with train/test class split...")

    gen = SyntheticGaussianMixture(
        n_classes_total=getattr(cfg.DATA, "N_CLASSES_TOTAL", 20),
        n_classes_train=getattr(cfg.DATA, "N_CLASSES_TRAIN", 15),
        modes_per_class=K_PROTOS,
        n_samples=getattr(cfg.DATA, "N_SAMPLES", 3000),
    )

    X_train, y_train, X_test, y_test = gen.generate(return_split=True)
    N_CLASSES = gen.n_classes_train

    train_loader = DataLoader(
        SimpleDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(cfg.DATA, "NUM_WORKERS", 0),
    )

    test_loader = DataLoader(
        SimpleDataset(X_test, y_test),
        batch_size=getattr(cfg.DATA, "TEST_BATCHSIZE", 128),
        shuffle=False,
        num_workers=getattr(cfg.DATA, "NUM_WORKERS", 0),
    )

    print(f"✓ Total classes: {gen.n_classes_total}")
    print(f"✓ Train classes (seen): {gen.n_classes_train}")
    print(f"✓ Test classes (unseen): {gen.n_classes_test}")
    print(f"✓ Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # ====================================================================
    # MODEL INITIALIZATION
    # ====================================================================
    print_banner("MODEL INITIALIZATION")

    model = EmbeddingNet(
        input_dim=cfg.MODEL.BACKBONE.INPUT_DIM,
        output_dim=EMBED_DIM,
    ).to(DEVICE)

    criterion = MpcbmlLoss(cfg).to(DEVICE)
    print("✓ Model initialized\n")

    # ====================================================================
    # PROTOTYPE INITIALIZATION
    # ====================================================================
    print_banner("PROTOTYPE INITIALIZATION")

    init_factory = PrototypeInitializerFactory()
    init_method = cfg.LOSSES.MPCBML_LOSS.INIT_METHOD

    print(f"Available methods: {init_factory.list_methods()}")
    print(f"Using method: {init_method}\n")

    initial_prototypes, cluster_sizes = init_factory.initialize(
        method=init_method,
        model=model,
        train_loader=train_loader,
        n_classes=N_CLASSES,
        prototype_per_class=K_PROTOS,
        embed_dim=EMBED_DIM,
        device=DEVICE,
    )
    criterion.set_prototypes_and_weights(initial_prototypes, cluster_sizes)

    print(f"✓ Prototypes initialized with '{init_method}' method")
    print(f"✓ Shape: {initial_prototypes.shape}\n")

    tracker = PrototypeTracker(N_CLASSES, K_PROTOS, EMBED_DIM)
    tracker.record(criterion.prototypes, epoch=0)

    # ====================================================================
    # OPTIMIZER SETUP
    # ====================================================================
    print_banner("OPTIMIZER SETUP")
    optimizer_main, optimizer_weights = build_optimizer(cfg, model, criterion)

    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    print_banner("STARTING TRAINING")

    history = {}  # single source of truth; _history_append will populate series

    snapshot_epochs = _make_snapshot_epochs(EPOCHS)

    for epoch in range(EPOCHS):
        current_epoch = epoch + 1

        model.train()
        criterion.train()

        total_loss = 0.0
        stat_sums = defaultdict(float)
        stat_count = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer_main.zero_grad()
            if optimizer_weights is not None:
                optimizer_weights.zero_grad()

            embeddings = model(batch_x)
            loss = criterion(embeddings, batch_y)
            loss.backward()

            # Your constrained update stays as-is (you stated it is correct)
            if hasattr(criterion, "constrained_weight_update"):
                criterion.constrained_weight_update()

            optimizer_main.step()
            if optimizer_weights is not None:
                optimizer_weights.step()

            total_loss += float(loss.item())

            batch_stats = criterion.get_last_stats()
            if batch_stats is not None:
                for k, v in batch_stats.items():
                    if _is_number(v):
                        stat_sums[k] += float(v)
                stat_count += 1

        tracker.record(criterion.prototypes, epoch=current_epoch)

        # --- Visualization snapshots at selected epochs ---
        if current_epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                X_embed, y_embed = compute_train_embeddings(model, train_loader, DEVICE)

                protos_now = flatten_protos(criterion.prototypes.detach().cpu())
                proto_labels = torch.arange(N_CLASSES).repeat_interleave(K_PROTOS)

                viz_logger.plot_embeddings_with_umap(
                    embeddings=X_embed.numpy(),
                    labels=y_embed.numpy(),
                    prototypes=protos_now.numpy(),
                    proto_labels=proto_labels.numpy(),
                    title=f"Epoch {current_epoch} - UMAP",
                    filename=f"epoch_{current_epoch:03d}_umap.png",
                )

        # --- Evaluation phase (recall@k) ---
        model.eval()
        with torch.no_grad():
            train_embs, train_targs = [], []
            for bx, by in train_loader:
                train_embs.append(model(bx.to(DEVICE)))
                train_targs.append(by.to(DEVICE))
            train_embs = torch.cat(train_embs)
            train_targs = torch.cat(train_targs)
            train_recalls = compute_recall_at_k(
                train_embs,
                train_targs,
                k_values=tuple(cfg.VALIDATION.RECALL_K),
            )

            val_embs, val_targs = [], []
            for bx, by in test_loader:
                val_embs.append(model(bx.to(DEVICE)))
                val_targs.append(by.to(DEVICE))
            val_embs = torch.cat(val_embs)
            val_targs = torch.cat(val_targs)
            val_recalls = compute_recall_at_k(
                val_embs,
                val_targs,
                k_values=tuple(cfg.VALIDATION.RECALL_K),
            )

        # --- Per-epoch record (THIS is what makes diagnostics meaningful) ---
        epoch_record = {}

        # Add aggregated internal stats
        if stat_count > 0:
            epoch_record.update({k: (v / stat_count) for k, v in stat_sums.items()})

        # Add “primary” epoch metrics
        mean_loss = total_loss / max(1, len(train_loader))
        epoch_record["loss"] = mean_loss

        # Make diagnostics work even if you only rely on epoch loss:
        epoch_record.setdefault("loss_total", mean_loss)

        epoch_record["train_r@1"] = float(train_recalls["R@1"])
        epoch_record["train_r@2"] = float(train_recalls["R@2"])
        epoch_record["train_r@4"] = float(train_recalls["R@4"])
        epoch_record["train_r@8"] = float(train_recalls["R@8"])
        epoch_record["val_r@1"] = float(val_recalls["R@1"])
        epoch_record["val_r@2"] = float(val_recalls["R@2"])
        epoch_record["val_r@4"] = float(val_recalls["R@4"])
        epoch_record["val_r@8"] = float(val_recalls["R@8"])

        _history_append(history, epoch_record, epoch_idx=current_epoch)

        # Persist each epoch (so you can inspect partial runs)
        _write_history_csv(history, out_csv_path=(output_dir / "training_history.csv"))

        # Print progress
        if current_epoch % cfg.VALIDATION.VERBOSE == 0:
            print(
                f"Epoch {current_epoch:03d}/{EPOCHS} | "
                f"Loss: {mean_loss:.4f} | "
                f"Train R@1: {train_recalls['R@1']:.4f} | "
                f"Val R@1: {val_recalls['R@1']:.4f}"
            )

    print("\n✓ Training complete!\n")

    # Plot diagnostics AFTER history is consistent and complete
    viz_logger.plot_loss_diagnostics(history, filename="06_loss_diagnostics.png")

    # ====================================================================
    # FINAL VISUALIZATIONS
    # ====================================================================
    print_banner("GENERATING VISUALIZATIONS")

    final_prototypes = criterion.prototypes.detach().cpu()
    X_embed, y_embed = compute_train_embeddings(model, train_loader, DEVICE)

    viz_config = getattr(cfg, "VISUALIZATION", None)
    if viz_config:
        save_prototypes_visualization(
            output_dir=str(output_dir),
            initial_prototypes=initial_prototypes,
            final_prototypes=final_prototypes,
            X_train=X_embed,
            y_train=y_embed,
            n_classes=N_CLASSES,
            tracker=tracker,
            save_initialization=getattr(viz_config, "SAVE_INITIALIZATION", True),
            save_movement=getattr(viz_config, "SAVE_MOVEMENT", True),
            save_trajectory=getattr(viz_config, "SAVE_TRAJECTORY", True),
            save_distance=getattr(viz_config, "SAVE_DISTANCE", True),
        )
    else:
        print("⚠️ VISUALIZATION config not found, skipping prototype visualizations")

    viz_logger.plot_training_metrics(history, "05_metrics.png")

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    print_banner("SAVING RESULTS")
    viz_logger.save_summary(history, args.config, "summary.txt")
    viz_logger.print_results(history)


if __name__ == "__main__":
    main()
