from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
import math
import umap
from sklearn.manifold import TSNE


def _to_numpy_unit(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float()
    x = np.asarray(x, dtype=np.float32)

    # Expect [N, D]
    if x.ndim != 2:
        raise ValueError(f"Expected [N, D], got shape {x.shape}")

    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class VisualizationLogger:
    """
    Manages all visualization and logging during training.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_loss_series(self, history: dict):
        # Prefer loss_total if present; fall back to loss
        if "loss_total" in history:
            return history["loss_total"]
        if "loss" in history:
            return history["loss"]
        raise KeyError("History must contain either 'loss_total' or 'loss'.")

    def plot_training_metrics(self, history: dict, filename: str = "metrics.png") -> None:
        """
        Plot training metrics:
        - Loss
        - Train vs Val Recall@1
        - Train vs Val Recall@2 & @4
        - All Recalls (Train vs Val)
        """
        if "epoch" not in history:
            raise KeyError("history['epoch'] is required for plotting.")

        epochs = history["epoch"]
        loss_series = self._get_loss_series(history)

        required = [
            "train_r@1", "train_r@2", "train_r@4", "train_r@8",
            "val_r@1", "val_r@2", "val_r@4", "val_r@8",
        ]
        missing = [k for k in required if k not in history]
        if missing:
            raise KeyError(f"Missing required history keys for metrics plot: {missing}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1) Loss
        axes[0, 0].plot(epochs, loss_series, "o-", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Epoch", fontsize=12)
        axes[0, 0].set_ylabel("Loss", fontsize=12)
        axes[0, 0].set_title("Training Loss", fontsize=13, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)

        # 2) Recall@1
        axes[0, 1].plot(epochs, history["train_r@1"], "o-", linewidth=2, markersize=6, label="Train R@1")
        axes[0, 1].plot(epochs, history["val_r@1"], "s--", linewidth=2, markersize=6, label="Val R@1")
        axes[0, 1].set_xlabel("Epoch", fontsize=12)
        axes[0, 1].set_ylabel("Recall@1", fontsize=12)
        axes[0, 1].set_title("Recall@1 (Train vs Val)", fontsize=13, fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3) Recall@2 & @4
        axes[1, 0].plot(epochs, history["train_r@2"], "o-", linewidth=2, markersize=6, label="Train R@2")
        axes[1, 0].plot(epochs, history["val_r@2"], "o--", linewidth=2, markersize=6, label="Val R@2")
        axes[1, 0].plot(epochs, history["train_r@4"], "s-", linewidth=2, markersize=6, label="Train R@4")
        axes[1, 0].plot(epochs, history["val_r@4"], "s--", linewidth=2, markersize=6, label="Val R@4")
        axes[1, 0].set_xlabel("Epoch", fontsize=12)
        axes[1, 0].set_ylabel("Recall", fontsize=12)
        axes[1, 0].set_title("Recall@2 & Recall@4", fontsize=13, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4) All recalls
        axes[1, 1].plot(epochs, history["train_r@1"], "o-", linewidth=2, markersize=6, label="Train R@1")
        axes[1, 1].plot(epochs, history["val_r@1"], "o--", linewidth=2, markersize=6, label="Val R@1")
        axes[1, 1].plot(epochs, history["train_r@2"], "s-", linewidth=2, markersize=6, label="Train R@2")
        axes[1, 1].plot(epochs, history["val_r@2"], "s--", linewidth=2, markersize=6, label="Val R@2")
        axes[1, 1].plot(epochs, history["train_r@4"], "^-", linewidth=2, markersize=6, label="Train R@4")
        axes[1, 1].plot(epochs, history["val_r@4"], "^--", linewidth=2, markersize=6, label="Val R@4")
        axes[1, 1].plot(epochs, history["train_r@8"], "D-", linewidth=2, markersize=6, label="Train R@8")
        axes[1, 1].plot(epochs, history["val_r@8"], "D--", linewidth=2, markersize=6, label="Val R@8")
        axes[1, 1].set_xlabel("Epoch", fontsize=12)
        axes[1, 1].set_ylabel("Recall", fontsize=12)
        axes[1, 1].set_title("All Recalls (Train vs Val)", fontsize=13, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f" ✓ Saved: {filename}")
        plt.close()

    def save_summary(self, history: dict, config_path: str, filename: str = "summary.txt") -> None:
        summary_path = self.output_dir / filename
        loss_series = self._get_loss_series(history)

        with open(summary_path, "w") as f:
            f.write("MP-CBML Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Best Loss: {min(loss_series):.6f}\n")
            f.write(f"Final Train R@1: {history['train_r@1'][-1]:.4f}\n")
            f.write(f"Final Val   R@1: {history['val_r@1'][-1]:.4f}\n")
            f.write(f"Final Train R@2: {history['train_r@2'][-1]:.4f}\n")
            f.write(f"Final Val   R@2: {history['val_r@2'][-1]:.4f}\n")
            f.write(f"Final Train R@4: {history['train_r@4'][-1]:.4f}\n")
            f.write(f"Final Val   R@4: {history['val_r@4'][-1]:.4f}\n")
            f.write(f"Final Train R@8: {history['train_r@8'][-1]:.4f}\n")
            f.write(f"Final Val   R@8: {history['val_r@8'][-1]:.4f}\n")

        print(f"   ✓ Saved: {filename}")

    def print_results(self, history: dict) -> None:
        loss_series = self._get_loss_series(history)

        print("\n📊 Final Results:")
        print(f"   Best Loss:  {min(loss_series):.6f}")
        print(f" Final Train R@1: {history['train_r@1'][-1]:.4f}")
        print(f" Final Val   R@1: {history['val_r@1'][-1]:.4f}")
        print(f" Final Train R@2: {history['train_r@2'][-1]:.4f}")
        print(f" Final Val   R@2: {history['val_r@2'][-1]:.4f}")
        print(f" Final Train R@4: {history['train_r@4'][-1]:.4f}")
        print(f" Final Val   R@4: {history['val_r@4'][-1]:.4f}")
        print(f" Final Train R@8: {history['train_r@8'][-1]:.4f}")
        print(f" Final Val   R@8: {history['val_r@8'][-1]:.4f}")
        print(f"📁 Output saved to: {self.output_dir}\n")

    def plot_embeddings_with_umap(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: Optional[np.ndarray] = None,
        proto_labels: Optional[np.ndarray] = None,
        filename: str = "embeddings_umap.png",
        title: str = "UMAP projection of embeddings",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        emb = _to_numpy_unit(embeddings)
        y = np.asarray(labels)

        proto = None
        if prototypes is not None:
            proto = _to_numpy_unit(prototypes)
            proto_y = np.asarray(proto_labels) if proto_labels is not None else None

            combined = np.vstack([emb, proto])
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            combined_2d = reducer.fit_transform(combined)
            emb_2d = combined_2d[: len(emb)]
            proto_2d = combined_2d[len(emb) :]
        else:
            proto_2d = None
            proto_y = None
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            emb_2d = reducer.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, s=8, alpha=0.35)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        if proto_2d is not None:
            if proto_y is not None:
                ax.scatter(
                    proto_2d[:, 0], proto_2d[:, 1],
                    c=proto_y, s=140, marker="*", edgecolors="black", linewidths=0.8
                )
            else:
                ax.scatter(
                    proto_2d[:, 0], proto_2d[:, 1],
                    s=140, marker="*", edgecolors="black", linewidths=0.8
                )

        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_embeddings_with_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: Optional[np.ndarray] = None,
        proto_labels: Optional[np.ndarray] = None,
        filename: str = "embeddings_tsne.png",
        title: str = "t-SNE projection of embeddings",
        perplexity: float = 30.0,
        random_state: int = 42,
    ):
        emb = _to_numpy_unit(embeddings)
        y = np.asarray(labels)

        proto = None
        if prototypes is not None:
            proto = _to_numpy_unit(prototypes)
            proto_y = np.asarray(proto_labels) if proto_labels is not None else None

            combined = np.vstack([emb, proto])
            tsne = TSNE(n_components=2, init="pca", random_state=random_state, perplexity=perplexity)
            combined_2d = tsne.fit_transform(combined)
            emb_2d = combined_2d[: len(emb)]
            proto_2d = combined_2d[len(emb) :]
        else:
            proto_2d = None
            proto_y = None
            tsne = TSNE(n_components=2, init="pca", random_state=random_state, perplexity=perplexity)
            emb_2d = tsne.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, s=8, alpha=0.35)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        if proto_2d is not None:
            if proto_y is not None:
                ax.scatter(
                    proto_2d[:, 0], proto_2d[:, 1],
                    c=proto_y, s=140, marker="*", edgecolors="black", linewidths=0.8
                )
            else:
                ax.scatter(
                    proto_2d[:, 0], proto_2d[:, 1],
                    s=140, marker="*", edgecolors="black", linewidths=0.8
                )

        ax.set_title(title)
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_loss_diagnostics(self, history: dict, filename: str = "06_loss_diagnostics.png"):
        """
        Plots MPCBML internal signals, if present in `history`.
        Requires `history["epoch"]` plus any diagnostic series you want.
        """
        if "epoch" not in history:
            return

        x = history["epoch"]

        # If you only logged "loss", still show it under the diagnostics umbrella.
        if "loss_total" not in history and "loss" in history and len(history["loss"]) == len(x):
            history = dict(history)  # shallow copy
            history["loss_total"] = history["loss"]

        candidates = [
            ("loss_total", "loss_total"),
            ("loss_main", "loss_main"),
            ("loss_reg", "loss_reg"),
            ("similarity_margin_mean", "s_pos - s_neg (mean)"),
            ("reg_violation_ratio", "P(s_neg > xi)"),
            ("xi_threshold", "xi"),
            ("s_pos_mean", "s_pos (mean)"),
            ("s_neg_mean", "s_neg (mean)"),
            ("proto_unused_ratio", "proto_unused_ratio"),
            ("weight_entropy_ratio", "weight_entropy_ratio"),
            ("weight_sum_violation_max", "max|sum(w)-1|"),
            ("embed_collapse_ratio", "embed_collapse_ratio"),
        ]

        series = [(k, label) for (k, label) in candidates if k in history and len(history[k]) == len(x)]
        if not series:
            return

        n = len(series)
        ncols = 3
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
        if nrows == 1:
            axes = np.array([axes])  # shape (1, ncols)
        axes = axes.reshape(nrows, ncols)

        for i, (k, label) in enumerate(series):
            r = i // ncols
            c = i % ncols
            ax = axes[r][c]
            ax.plot(x, history[k])
            ax.set_title(label)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)

        for j in range(n, nrows * ncols):
            r = j // ncols
            c = j % ncols
            axes[r][c].axis("off")

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def save_prototypes_visualization(
    output_dir: str,
    initial_prototypes: torch.Tensor,
    final_prototypes: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_classes: int,
    tracker=None,
    save_initialization: bool = True,
    save_movement: bool = True,
    save_trajectory: bool = True,
    save_distance: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from core.utils.prototype_logger import (
        plot_prototype_initialization,
        plot_prototype_movement,
        plot_prototype_trajectory,
        plot_movement_distance,
        plot_data_and_prototypes,
    )

    print("\n" + "=" * 70)
    print("📈 GENERATING PROTOTYPE VISUALIZATIONS")
    print("=" * 70 + "\n")

    if save_initialization:
        print("1️⃣  Prototype initialization...")
        fig1 = plot_prototype_initialization(
            initial_prototypes.cpu(),
            initial_kmeans=None,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
        )
        if fig1:
            plt.savefig(output_path / "01_initialization.png", dpi=150, bbox_inches="tight")
            print("   ✓ Saved: 01_initialization.png")
            plt.close()

    if save_movement:
        print("2️⃣  Prototype movement...")
        fig2 = plot_prototype_movement(
            initial_prototypes.cpu(),
            final_prototypes,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
        )
        if fig2:
            plt.savefig(output_path / "02_movement.png", dpi=150, bbox_inches="tight")
            print("   ✓ Saved: 02_movement.png")
            plt.close()

    if save_trajectory and tracker is not None:
        print("3️⃣  Prototype trajectory...")
        fig3 = plot_prototype_trajectory(
            tracker,
            initial_prototypes.cpu(),
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
            sample_epochs=5,
        )
        if fig3:
            plt.savefig(output_path / "03_trajectory.png", dpi=150, bbox_inches="tight")
            print("   ✓ Saved: 03_trajectory.png")
            plt.close()

    if save_distance and tracker is not None:
        print("4️⃣  Movement distance...")
        fig4 = plot_movement_distance(tracker, initial_prototypes.cpu())
        if fig4:
            plt.savefig(output_path / "04_distance.png", dpi=150, bbox_inches="tight")
            print("   ✓ Saved: 04_distance.png")
            plt.close()

    # 5) Data + prototypes for multiple epochs (initial + selected)
    if tracker is not None:
        epochs_to_plot = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for ep in epochs_to_plot:
            if ep in tracker.epoch_numbers:
                idx = tracker.epoch_numbers.index(ep)
                protos_ep = tracker.history[idx]  # [C, K, D]
                fig_ep = plot_data_and_prototypes(
                    X_train=X_train,
                    y_train=y_train,
                    protos=protos_ep,
                    n_classes=n_classes,
                    title=f"Epoch {ep}",
                )
                if fig_ep:
                    plt.savefig(output_path / f"epoch_{ep:03d}_data_protos.png", dpi=150, bbox_inches="tight")
                    print(f" ✓ Saved: epoch_{ep:03d}_data_protos.png")
                    plt.close()

    print()
