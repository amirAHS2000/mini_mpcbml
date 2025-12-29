import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_data_and_prototypes(X_train, y_train, protos, n_classes=4, title=""):
    """
    Plot data distribution + prototypes for a given epoch (2D only).

    X_train: [N, 2] tensor or numpy (on CPU)
    y_train: [N]
    protos:  [C, K, 2] tensor (on CPU)
    """
    if protos.shape[-1] != 2:
        print(f"⚠️ Skipping 2D visualization: embedding dim {protos.shape[-1]} != 2")
        return None

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    if isinstance(X_train, torch.Tensor):
        X_np = X_train.cpu().numpy()
        y_np = y_train.cpu().numpy()
    else:
        X_np, y_np = X_train, y_train

    C, K = protos.shape[0], protos.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.3)

    # Data clusters
    for c in range(n_classes):
        class_data = X_np[y_np == c]
        ax.scatter(class_data[:, 0], class_data[:, 1],
                   alpha=0.15, s=10, color=colors[c])

    # Prototypes
    protos_np = protos.cpu().numpy()
    for c in range(C):
        for k in range(K):
            p = protos_np[c, k]
            ax.scatter(p[0], p[1], s=200, marker='*',
                       color=colors[c], edgecolors='black', linewidth=2)

    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def plot_prototype_initialization(initial_random, initial_kmeans=None,
                                   X_train=None, y_train=None, n_classes=4):
    """
    Visualize random initialization vs K-means initialization.

    Args:
        initial_random: [C, K, D] random prototypes
        initial_kmeans: [C, K, D] K-means prototypes (optional)
        X_train: [N, 2] training data (for 2D visualization)
        y_train: [N] training labels
        n_classes: number of classes
    """
    if initial_random.shape[-1] != 2:
        print(f"⚠️  Skipping 2D visualization: embedding dimension is {initial_random.shape[-1]}, not 2")
        print("   (Only works with 2D embeddings)")
        return None

    C, K = initial_random.shape[0], initial_random.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    num_plots = 3 if initial_kmeans is not None and X_train is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot 1: Random Initialization
    ax = axes[plot_idx]
    ax.set_title('Initial Prototypes (Embedding Space)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.3)

    for c in range(C):
        protos_c = initial_random[c]
        ax.scatter(protos_c[:, 0], protos_c[:, 1],
                  s=200, marker='*', color=colors[c],
                  edgecolors='black', linewidth=2,
                  label=f'Class {c}', zorder=5)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    plot_idx += 1

    # Plot 2: Data Distribution (if available)
    if X_train is not None and y_train is not None:
        ax = axes[plot_idx]
        ax.set_title('📊 Data Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)

        for c in range(n_classes):
            class_data = X_train[y_train == c]
            ax.scatter(class_data[:, 0], class_data[:, 1],
                      alpha=0.3, s=20, color=colors[c], label=f'Class {c}')
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
        plot_idx += 1

    # Plot 3: K-Means Initialization (if available)
    if initial_kmeans is not None:
        ax = axes[plot_idx]
        ax.set_title('🎯 K-Means Initialization (Optional)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)

        for c in range(C):
            protos_c = initial_kmeans[c]
            ax.scatter(protos_c[:, 0], protos_c[:, 1],
                      s=200, marker='s', color=colors[c],
                      edgecolors='black', linewidth=2,
                      label=f'Class {c}', zorder=5)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig

def plot_prototype_movement(initial_protos, final_protos, X_train=None, y_train=None,
                            n_classes=4, title_suffix=""):
    """
    Visualize movement of prototypes from initial to final state.

    Args:
        initial_protos: [C, K, D] initial prototypes
        final_protos: [C, K, D] final trained prototypes
        X_train: [N, 2] training data (optional)
        y_train: [N] training labels (optional)
        n_classes: number of classes
        title_suffix: suffix for title
    """
    if initial_protos.shape[-1] != 2:
        print(f"⚠️  Skipping 2D visualization: embedding dimension is {initial_protos.shape[-1]}, not 2")
        return None

    C, K = initial_protos.shape[0], initial_protos.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot data distribution if available
    if X_train is not None and y_train is not None:
        for c in range(n_classes):
            class_data = X_train[y_train == c]
            ax.scatter(class_data[:, 0], class_data[:, 1],
                      alpha=0.15, s=10, color=colors[c])

    # Plot prototype movement
    for c in range(C):
        for k in range(K):
            init_proto = initial_protos[c, k].numpy() if isinstance(initial_protos[c, k], torch.Tensor) else initial_protos[c, k]
            final_proto = final_protos[c, k].numpy() if isinstance(final_protos[c, k], torch.Tensor) else final_protos[c, k]

            # Arrow from initial to final
            ax.annotate('', xy=final_proto, xytext=init_proto,
                       arrowprops=dict(arrowstyle='->', lw=2, color=colors[c], alpha=0.6))

            # Initial position
            ax.scatter(*init_proto, s=150, marker='o', color=colors[c],
                      edgecolors='black', linewidth=1.5, alpha=0.6, zorder=4)

            # Final position
            ax.scatter(*final_proto, s=250, marker='*', color=colors[c],
                      edgecolors='black', linewidth=2, zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(color='lightgray', label='○ Initial (Random)'),
        mpatches.Patch(color='lightgray', label='★ Final (Trained)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    ax.set_title(f'🚀 Prototype Movement: Random → Trained{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig

def plot_prototype_trajectory(tracker, initial_protos, X_train=None, y_train=None,
                              n_classes=4, sample_epochs=5):
    """
    Show prototype evolution across selected epochs.

    Args:
        tracker: PrototypeTracker object
        initial_protos: [C, K, D] initial prototypes
        X_train: [N, 2] training data (optional)
        y_train: [N] training labels (optional)
        n_classes: number of classes
        sample_epochs: number of epochs to sample for visualization
    """
    if initial_protos.shape[-1] != 2:
        print(f"⚠️  Skipping 2D visualization: embedding dimension is {initial_protos.shape[-1]}, not 2")
        return None

    C, K = initial_protos.shape[0], initial_protos.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    # Select epochs to visualize
    total_epochs = len(tracker.history)
    if total_epochs <= sample_epochs:
        selected_indices = list(range(total_epochs))
    else:
        selected_indices = [int(i * (total_epochs - 1) / (sample_epochs - 1))
                           for i in range(sample_epochs)]

    fig, axes = plt.subplots(1, sample_epochs + 1, figsize=(4*(sample_epochs+1), 4))
    if sample_epochs == 0:
        axes = [axes]

    # Plot 0: Initial
    ax = axes[0]
    ax.set_title('Initial (Random)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.3)

    if X_train is not None and y_train is not None:
        for c in range(n_classes):
            class_data = X_train[y_train == c]
            ax.scatter(class_data[:, 0], class_data[:, 1], alpha=0.1, s=5, color=colors[c])

    for c in range(C):
        protos = initial_protos[c]
        ax.scatter(protos[:, 0], protos[:, 1], s=150, marker='o',
                  color=colors[c], edgecolors='black', linewidth=1, zorder=4)
    ax.set_aspect('equal')

    # Plot selected epochs
    for idx, epoch_idx in enumerate(selected_indices):
        ax = axes[idx + 1]
        epoch_num = tracker.epoch_numbers[epoch_idx]

        ax.set_title(f'Epoch {epoch_num}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)

        if X_train is not None and y_train is not None:
            for c in range(n_classes):
                class_data = X_train[y_train == c]
                ax.scatter(class_data[:, 0], class_data[:, 1], alpha=0.1, s=5, color=colors[c])

        protos = tracker.history[epoch_idx]
        for c in range(C):
            protos_c = protos[c]
            ax.scatter(protos_c[:, 0], protos_c[:, 1], s=150, marker='o',
                      color=colors[c], edgecolors='black', linewidth=1, zorder=4)
        ax.set_aspect('equal')

    plt.suptitle('📈 Prototype Evolution Over Training', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_movement_distance(tracker, initial_protos):
    """
    Plot average prototype movement distance over epochs.

    Args:
        tracker: PrototypeTracker object
        initial_protos: [C, K, D] initial prototypes
    """
    movements = tracker.get_movement(initial_protos)
    epochs = tracker.epoch_numbers

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, movements, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(epochs, movements, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average L2 Distance from Initial', fontsize=12, fontweight='bold')
    ax.set_title('📏 Prototype Movement Distance Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig