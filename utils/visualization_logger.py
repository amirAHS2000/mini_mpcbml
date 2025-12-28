import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
import torch


class VisualizationLogger:
    """
    Manages all visualization and logging during training.
    Separates visualization logic from training loop.
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
        Plot training metrics (loss and recall curves).
        
        Args:
            history: Dictionary with keys 'epoch', 'loss', 'r@1', 'r@2', 'r@4', 'r@8'
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curve
        axes[0, 0].plot(history['epoch'], history['loss'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall@1
        axes[0, 1].plot(history['epoch'], history['r@1'], 'o-', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Recall@1', fontsize=12)
        axes[0, 1].set_title('Recall@1', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall@2 & @4
        axes[1, 0].plot(history['epoch'], history['r@2'], 'o-', linewidth=2, markersize=6, label='Recall@2')
        axes[1, 0].plot(history['epoch'], history['r@4'], 's-', linewidth=2, markersize=6, label='Recall@4')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Recall', fontsize=12)
        axes[1, 0].set_title('Recall@K', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # All Recalls
        axes[1, 1].plot(history['epoch'], history['r@1'], 'o-', linewidth=2, markersize=6, label='R@1')
        axes[1, 1].plot(history['epoch'], history['r@2'], 's-', linewidth=2, markersize=6, label='R@2')
        axes[1, 1].plot(history['epoch'], history['r@4'], '^-', linewidth=2, markersize=6, label='R@4')
        axes[1, 1].plot(history['epoch'], history['r@8'], 'D-', linewidth=2, markersize=6, label='R@8')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('All Recall Metrics', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
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
            f.write(f"Final R@1: {history['r@1'][-1]:.4f}\n")
            f.write(f"Final R@2: {history['r@2'][-1]:.4f}\n")
            f.write(f"Final R@4: {history['r@4'][-1]:.4f}\n")
            f.write(f"Final R@8: {history['r@8'][-1]:.4f}\n")
        
        print(f"   ✓ Saved: {filename}")
    
    def print_results(self, history: dict) -> None:
        """
        Print final training results.
        
        Args:
            history: Training history dictionary
        """
        print(f"\n📊 Final Results:")
        print(f"   Best Loss:  {min(history['loss']):.6f}")
        print(f"   Final R@1:  {history['r@1'][-1]:.4f}")
        print(f"   Final R@2:  {history['r@2'][-1]:.4f}")
        print(f"   Final R@4:  {history['r@4'][-1]:.4f}")
        print(f"   Final R@8:  {history['r@8'][-1]:.4f}\n")
        
        print(f"📁 Output saved to: {self.output_dir}\n")


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
    from utils.prototype_logger import (
        plot_prototype_initialization,
        plot_prototype_movement,
        plot_prototype_trajectory,
        plot_movement_distance
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
    
    print()