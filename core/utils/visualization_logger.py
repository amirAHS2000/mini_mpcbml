import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np


class VisualizationLogger:
    """
    Manages all visualization and logging during training.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize visualization logger.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_metrics(self, history: dict, filename: str = 'metrics.png') -> None:
        """
        Plot training metrics:
        - Loss
        - Train vs Val Recall@1
        - Train vs Val Recall@2 & @4
        - All Recalls (Train vs Val)
        Expected keys:
        'epoch', 'loss',
        'train_r@1', 'train_r@2', 'train_r@4', 'train_r@8',
        'val_r@1',   'val_r@2',   'val_r@4',   'val_r@8'
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = history['epoch']

        # 1) Loss (train loss only)
        axes[0, 0].plot(epochs, history['loss'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 2) Recall@1: Train vs Val
        axes[0, 1].plot(epochs, history['train_r@1'], 'o-', linewidth=2, markersize=6,
                        color='blue', label='Train R@1')
        axes[0, 1].plot(epochs, history['val_r@1'], 's--', linewidth=2, markersize=6,
                        color='orange', label='Val R@1')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Recall@1', fontsize=12)
        axes[0, 1].set_title('Recall@1 (Train vs Val)', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3) Recall@2 & @4: Train vs Val
        axes[1, 0].plot(epochs, history['train_r@2'], 'o-', linewidth=2, markersize=6,
                        label='Train R@2')
        axes[1, 0].plot(epochs, history['val_r@2'], 'o--', linewidth=2, markersize=6,
                        label='Val R@2')
        axes[1, 0].plot(epochs, history['train_r@4'], 's-', linewidth=2, markersize=6,
                        label='Train R@4')
        axes[1, 0].plot(epochs, history['val_r@4'], 's--', linewidth=2, markersize=6,
                        label='Val R@4')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Recall', fontsize=12)
        axes[1, 0].set_title('Recall@2 & Recall@4', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4) All Recalls: Train vs Val
        axes[1, 1].plot(epochs, history['train_r@1'], 'o-',  linewidth=2, markersize=6, label='Train R@1')
        axes[1, 1].plot(epochs, history['val_r@1'],   'o--', linewidth=2, markersize=6, label='Val R@1')
        axes[1, 1].plot(epochs, history['train_r@2'], 's-',  linewidth=2, markersize=6, label='Train R@2')
        axes[1, 1].plot(epochs, history['val_r@2'],   's--', linewidth=2, markersize=6, label='Val R@2')
        axes[1, 1].plot(epochs, history['train_r@4'], '^-',  linewidth=2, markersize=6, label='Train R@4')
        axes[1, 1].plot(epochs, history['val_r@4'],   '^--', linewidth=2, markersize=6, label='Val R@4')
        axes[1, 1].plot(epochs, history['train_r@8'], 'D-',  linewidth=2, markersize=6, label='Train R@8')
        axes[1, 1].plot(epochs, history['val_r@8'],   'D--', linewidth=2, markersize=6, label='Val R@8')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('All Recalls (Train vs Val)', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" ✓ Saved: {filename}")
        plt.close()

    
    def save_summary(self, history: dict, config_path: str, filename: str = 'summary.txt') -> None:
        """
        Save training summary to text file.
        
        Args:
            history: Training history dictionary
            config_path: Path to config file used
            filename: Output filename
        """
        summary_path = self.output_dir / filename
        
        with open(summary_path, 'w') as f:
            f.write("MP-CBML Training Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Best Loss: {min(history['loss']):.6f}\n")
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
        """
        Print final training results.
        
        Args:
            history: Training history dictionary
        """
        print(f"\n📊 Final Results:")
        print(f"   Best Loss:  {min(history['loss']):.6f}")
        print(f" Final Train R@1: {history['train_r@1'][-1]:.4f}")
        print(f" Final Val   R@1: {history['val_r@1'][-1]:.4f}")
        print(f" Final Train R@2: {history['train_r@2'][-1]:.4f}")
        print(f" Final Val   R@2: {history['val_r@2'][-1]:.4f}")
        print(f" Final Train R@4: {history['train_r@4'][-1]:.4f}")
        print(f" Final Val   R@4: {history['val_r@4'][-1]:.4f}")
        print(f" Final Train R@8: {history['train_r@8'][-1]:.4f}")
        print(f" Final Val   R@8: {history['val_r@8'][-1]:.4f}")
        
        print(f"📁 Output saved to: {self.output_dir}\n")

    def plot_embeddings_with_umap(self, embeddings, labels, prototypes=None,
                                   proto_labels=None, title="Embeddings (UMAP)",
                                   filename="embeddings_umap.png"):
        """
        Visualize high-dimensional embeddings using UMAP projection.
        
        Args:
            embeddings: [N, embed_dim] tensor or array
            labels: [N] class labels
            prototypes: [K*M, embed_dim] prototype positions (optional)
            proto_labels: [K*M] prototype class labels (optional)
            title: Plot title
            filename: Save filename
        """
        try:
            import umap
        except ImportError:
            print("⚠️ UMAP not installed. Install with: pip install umap-learn")
            return
        
        # Convert to numpy
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        
        # Project to 2D
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data points
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='tab20', alpha=0.6, s=20, label='Data'
        )
        
        # Plot prototypes if provided
        if prototypes is not None:
            if hasattr(prototypes, 'cpu'):
                prototypes = prototypes.cpu().numpy()
            prototypes = np.asarray(prototypes, dtype=np.float32)
            proto_labels = np.asarray(proto_labels, dtype=np.int64)
            
            # Project prototypes using same UMAP transform
            # (Note: UMAP doesn't support direct projection of new points,
            # so we'll refit with combined data for consistency)
            combined = np.vstack([embeddings, prototypes])
            combined_2d = reducer.fit_transform(combined)
            
            proto_2d = combined_2d[len(embeddings):]
            
            ax.scatter(
                proto_2d[:, 0], proto_2d[:, 1],
                c=proto_labels, cmap='tab20', marker='*', 
                s=800, edgecolors='black', linewidths=2,
                label='Prototypes', alpha=0.9
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(self.output_dir) / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filename}")


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
    save_distance: bool = True
) -> None:
    """
    Save prototype-related visualizations.
    
    This is a wrapper that integrates with your existing
    plot_prototype_* functions from prototype_logger.py
    
    Args:
        output_dir: Directory to save visualizations
        initial_prototypes: Initial prototype embeddings
        final_prototypes: Final trained prototype embeddings
        X_train: Training data
        y_train: Training labels
        n_classes: Number of classes
        tracker: Prototype tracker for trajectory visualization
        save_initialization: Whether to save initialization plot
        save_movement: Whether to save movement plot
        save_trajectory: Whether to save trajectory plot
        save_distance: Whether to save distance plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Import visualization functions (from utils.prototype_logger)
    from core.utils.prototype_logger import (
        plot_prototype_initialization,
        plot_prototype_movement,
        plot_prototype_trajectory,
        plot_movement_distance,
        plot_data_and_prototypes,
    )
    
    print("\n" + "="*70)
    print("📈 GENERATING PROTOTYPE VISUALIZATIONS")
    print("="*70 + "\n")
    
    # 1. Initialization
    if save_initialization:
        print("1️⃣  Prototype initialization...")
        fig1 = plot_prototype_initialization(
            initial_prototypes.cpu(),
            initial_kmeans=None,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes
        )
        if fig1:
            plt.savefig(output_path / '01_initialization.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved: 01_initialization.png")
            plt.close()
    
    # 2. Movement
    if save_movement:
        print("2️⃣  Prototype movement...")
        fig2 = plot_prototype_movement(
            initial_prototypes.cpu(),
            final_prototypes,
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes
        )
        if fig2:
            plt.savefig(output_path / '02_movement.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved: 02_movement.png")
            plt.close()
    
    # 3. Trajectory
    if save_trajectory and tracker is not None:
        print("3️⃣  Prototype trajectory...")
        fig3 = plot_prototype_trajectory(
            tracker,
            initial_prototypes.cpu(),
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
            sample_epochs=5
        )
        if fig3:
            plt.savefig(output_path / '03_trajectory.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved: 03_trajectory.png")
            plt.close()
    
    # 4. Distance
    if save_distance and tracker is not None:
        print("4️⃣  Movement distance...")
        fig4 = plot_movement_distance(tracker, initial_prototypes.cpu())
        if fig4:
            plt.savefig(output_path / '04_distance.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved: 04_distance.png")
            plt.close()

        # 5. Data + prototypes for multiple epochs (initial + selected)
    if tracker is not None:
        # choose which epochs to visualize
        epochs_to_plot = [0, 1, 5, 10, 15, 20, 30, 40, 50]
        for ep in epochs_to_plot:
            if ep in tracker.epoch_numbers:
                idx = tracker.epoch_numbers.index(ep)
                protos_ep = tracker.history[idx]   # [C, K, D]
                fig_ep = plot_data_and_prototypes(
                    X_train=X_train,
                    y_train=y_train,
                    protos=protos_ep,
                    n_classes=n_classes,
                    title=f"Epoch {ep}"
                )
                if fig_ep:
                    plt.savefig(
                        output_path / f"epoch_{ep:03d}_data_protos.png",
                        dpi=150, bbox_inches='tight'
                    )
                    print(f" ✓ Saved: epoch_{ep:03d}_data_protos.png")
                    plt.close()

    print()