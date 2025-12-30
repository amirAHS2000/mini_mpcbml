import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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


def print_banner(text: str):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(f"🚀 {text}")
    print("="*70 + "\n")

def compute_train_embeddings(model, train_loader, device):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            feats = model(x)          # [B, embed_dim]
            all_feats.append(feats.cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

# Helper: ensure prototypes are 2D [C*K, D]
def flatten_protos(protos: torch.Tensor) -> torch.Tensor:
    # protos can be [C, K, D] or [C*K, D]
    if protos.dim() == 3:
        return protos.view(-1, protos.shape[-1])
    return protos

def main():
    # ====================================================================
    # ARGUMENT PARSING & CONFIGURATION
    # ====================================================================
    parser = argparse.ArgumentParser(
        description='MP-CBML Training with YAML Configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mpcbml_toy.yaml',
        help='Path to YAML config file'
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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    output_dir = cfg.OUTPUT.SAVE_DIR
    viz_logger = VisualizationLogger(output_dir)
    print(f"✓ Visualization logger initialized")
    print(f"✓ Output directory: {output_dir}\n")
    
    # ====================================================================
    # DATA PREPARATION
    # ====================================================================
    print_banner("DATA PREPARATION")
    
    print("Generating synthetic dataset with train/test class split...")

    gen = SyntheticGaussianMixture(
        n_classes_total=getattr(cfg.DATA, 'N_CLASSES_TOTAL', 20),
        n_classes_train=getattr(cfg.DATA, 'N_CLASSES_TRAIN', 15),
        modes_per_class=K_PROTOS,
        n_samples=getattr(cfg.DATA, 'N_SAMPLES', 3000)
    )

    X_train, y_train, X_test, y_test = gen.generate(return_split=True)

    N_CLASSES = gen.n_classes_train

    train_loader = DataLoader(
        SimpleDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(cfg.DATA, 'NUM_WORKERS', 0)
    )

    test_loader = DataLoader(
        SimpleDataset(X_test, y_test),
        batch_size=getattr(cfg.DATA, 'TEST_BATCHSIZE', 128),
        shuffle=False,
        num_workers=getattr(cfg.DATA, 'NUM_WORKERS', 0)
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
        output_dim=EMBED_DIM
    ).to(DEVICE)
    
    criterion = MpcbmlLoss(cfg).to(DEVICE)
    print(f"✓ Model initialized\n")
    
    # ====================================================================
    # PROTOTYPE INITIALIZATION WITH FACTORY PATTERN
    # ====================================================================
    print_banner("PROTOTYPE INITIALIZATION")
    
    # Create factory
    init_factory = PrototypeInitializerFactory()
    
    # Get initialization method from config
    init_method = cfg.LOSSES.MPCBML_LOSS.INIT_METHOD
    
    print(f"Available methods: {init_factory.list_methods()}")
    print(f"Using method: {init_method}\n")
    
    # Initialize prototypes using selected method
    initial_prototypes, cluster_sizes = init_factory.initialize(
        method=init_method,
        model=model,
        train_loader=train_loader,
        n_classes=N_CLASSES,
        prototype_per_class=K_PROTOS,
        embed_dim=EMBED_DIM,
        device=DEVICE
    )
    criterion.set_prototypes_and_weights(initial_prototypes, cluster_sizes)
    
    print(f"✓ Prototypes initialized with '{init_method}' method")
    print(f"✓ Shape: {initial_prototypes.shape}\n")
    
    # Create prototype tracker
    tracker = PrototypeTracker(N_CLASSES, K_PROTOS, EMBED_DIM)
    tracker.record(criterion.prototypes, epoch=0)
    
    # ====================================================================
    # OPTIMIZER & SCHEDULER SETUP
    # ==================================================================== 
    print_banner("OPTIMIZER SETUP")
   
    optimizer_main, optimizer_weights = build_optimizer(cfg, model, criterion)
        
    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    print_banner("STARTING TRAINING")
    history = {
        'epoch': [],
        'loss': [],
        'train_r@1': [],
        'train_r@2': [],
        'train_r@4': [],
        'train_r@8': [],
        'val_r@1': [],
        'val_r@2': [],
        'val_r@4': [],
        'val_r@8': [],
    }

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        criterion.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer_main.zero_grad()
            if optimizer_weights is not None:
                optimizer_weights.zero_grad()

            embeddings = model(batch_x)
            loss = criterion(embeddings, batch_y)
            loss.backward()

            if hasattr(criterion, 'constrained_weight_update'):
                criterion.constrained_weight_update()

            optimizer_main.step()
            if optimizer_weights is not None:
                optimizer_weights.step()

            total_loss += loss.item()

        tracker.record(criterion.prototypes, epoch=epoch+1)

        # --- Visualization snapshots at selected epochs ---
        snapshot_epochs = [0, EPOCHS // 4, EPOCHS // 2, 3 * EPOCHS // 4, EPOCHS]
        current_epoch = epoch + 1

        if current_epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                # Compute full train embeddings for this epoch
                X_embed, y_embed = compute_train_embeddings(model, train_loader, DEVICE)

                # Current prototypes
                protos_now = flatten_protos(criterion.prototypes.detach().cpu())
                proto_labels = torch.arange(N_CLASSES).repeat_interleave(K_PROTOS)

                # Decide filename/title per epoch
                if current_epoch == 1:
                    title = f"Epoch {current_epoch} (near init) - UMAP"
                    fname = f"epoch_{current_epoch:03d}_umap.png"
                elif current_epoch == EPOCHS:
                    title = f"Epoch {current_epoch} (final) - UMAP"
                    fname = f"epoch_{current_epoch:03d}_umap.png"
                else:
                    title = f"Epoch {current_epoch} - UMAP"
                    fname = f"epoch_{current_epoch:03d}_umap.png"

                viz_logger.plot_embeddings_with_umap(
                    embeddings=X_embed,
                    labels=y_embed,
                    prototypes=protos_now,
                    proto_labels=proto_labels,
                    title=title,
                    filename=fname,
                )

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            train_embs, train_targs = [], []
            for bx, by in train_loader:
                train_embs.append(model(bx.to(DEVICE)))
                train_targs.append(by.to(DEVICE))
            train_embs = torch.cat(train_embs)
            train_targs = torch.cat(train_targs)
            train_recalls = compute_recall_at_k(
                train_embs, train_targs,
                k_values=tuple(cfg.VALIDATION.RECALL_K)
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
                k_values=tuple(cfg.VALIDATION.RECALL_K)
            )
        

        stats = criterion.get_last_stats()

        # Store metrics
        history['epoch'].append(epoch+1)
        history['loss'].append(total_loss / len(train_loader))
        history['train_r@1'].append(train_recalls['R@1'])
        history['train_r@2'].append(train_recalls['R@2'])
        history['train_r@4'].append(train_recalls['R@4'])
        history['train_r@8'].append(train_recalls['R@8'])
        history['val_r@1'].append(val_recalls['R@1'])
        history['val_r@2'].append(val_recalls['R@2'])
        history['val_r@4'].append(val_recalls['R@4'])
        history['val_r@8'].append(val_recalls['R@8'])

        # Print progress
        if (epoch + 1) % cfg.VALIDATION.VERBOSE == 0:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train R@1: {train_recalls['R@1']:.4f} | "
                  f"Val R@1: {val_recalls['R@1']:.4f}")

    print("\n✓ Training complete!\n")
    
    # ====================================================================
    # VISUALIZATIONS USING SEPARATED LOGGER
    # ====================================================================
    print_banner("GENERATING VISUALIZATIONS")

    # Compute final embeddings (only once!)
    final_prototypes = criterion.prototypes.detach().cpu()
    X_embed, y_embed = compute_train_embeddings(model, train_loader, DEVICE)

    # ✅ FIX 2: Correct numbering for prototype visualizations
    print("\n2️⃣ Saving prototype visualizations (initialization, movement, trajectory, distance)...")
    viz_config = getattr(cfg, 'VISUALIZATION', None)
    if viz_config:
        # ✅ FIX 3: Pass embedding space data, not input space
        save_prototypes_visualization(
            output_dir=output_dir,
            initial_prototypes=initial_prototypes,
            final_prototypes=final_prototypes,
            X_train=X_embed,  # ← Now correctly in embedding space
            y_train=y_embed,  # ← Labels (same as before)
            n_classes=N_CLASSES,
            tracker=tracker,
            save_initialization=getattr(viz_config, 'SAVE_INITIALIZATION', True),
            save_movement=getattr(viz_config, 'SAVE_MOVEMENT', True),
            save_trajectory=getattr(viz_config, 'SAVE_TRAJECTORY', True),
            save_distance=getattr(viz_config, 'SAVE_DISTANCE', True)
        )
    else:
        print("⚠️ VISUALIZATION config not found, skipping prototype visualizations")

    # ✅ FIX 4: Correct numbering for metrics
    print("\n5️⃣ Plotting training metrics...")
    viz_logger.plot_training_metrics(history, '05_metrics.png')
    
    # ====================================================================
    # SAVE RESULTS USING LOGGER
    # ====================================================================
    print_banner("SAVING RESULTS")
    
    viz_logger.save_summary(history, args.config, 'summary.txt')
    viz_logger.print_results(history)


if __name__ == '__main__':
    main()
