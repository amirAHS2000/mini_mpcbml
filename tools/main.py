import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import matplotlib.pyplot as plt

from data.base_dataset import SimpleDataset
from data.data_generator import SyntheticGaussianMixture
from modeling.backbone.simple_embedding_net import EmbeddingNet
from losses.mpcbml_loss import MpcbmlLoss
from data.evaluation.retrieval_metric import compute_recall_at_k
from utils.prototype_tracker import PrototypeTracker
from utils.prototype_logger import (
    plot_prototype_initialization,
    plot_prototype_movement,
    plot_movement_distance,
    plot_prototype_trajectory
)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 MPCBML TRAINING WITH PROTOTYPE MOVEMENT VISUALIZATION")
    print("="*70 + "\n")

    # Settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_CLASSES = 4
    K_PROTOS = 3
    EMBED_DIM = 64
    BATCH_SIZE = 128
    LR = 0.001
    EPOCHS = 20

    # Data
    print("📊 Generating synthetic dataset...")
    gen = SyntheticGaussianMixture(n_classes=N_CLASSES, modes_per_class=K_PROTOS, n_samples=3000)
    X, y = gen.generate()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_loader = DataLoader(SimpleDataset(X_train, y_train),
                             batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SimpleDataset(X_test, y_test),
                            batch_size=BATCH_SIZE, shuffle=False)
    print(f"✓ Dataset created: {len(X_train)} training, {len(X_test)} test samples\n")

    # Model & Loss
    model = EmbeddingNet(input_dim=2, output_dim=EMBED_DIM).to(DEVICE)

    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(DEVICE=DEVICE, HEAD=SimpleNamespace(DIM=EMBED_DIM)),
        LOSSES=SimpleNamespace(
            MPCBML_LOSS=SimpleNamespace(
                N_CLASSES=N_CLASSES,
                PROTOTYPE_PER_CLASS=K_PROTOS,
                MA_MOMENTUM=0.9,
                GAMMA_REG=0.5,
                LAMBDA_REG=0.1,
                THETA_IS_LEARNABLE=True,
                INIT_THETA=2.0
            )
        )
    )

    criterion = MpcbmlLoss(cfg).to(DEVICE)

    # IMPORTANT: Initialize with random prototypes (not K-means)
    print("🎲 Initializing prototypes with RANDOM initialization...\n")
    initial_prototypes = initialize_mpcbml(
        model, train_loader, criterion, DEVICE,
        use_random_init=True,  # ✅ Random initialization
        use_kmeans=False         # Skip K-means refinement
    )

    # Create prototype tracker
    tracker = PrototypeTracker(N_CLASSES, K_PROTOS, EMBED_DIM)
    tracker.record(criterion.prototypes, epoch=0)

    # Optimizer
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': LR * 10}
    ], lr=LR)

    # Training Loop
    print("🎯 Starting training...\n")
    history = {'epoch': [], 'loss': [], 'r@1': [], 'r@2': []}

    for epoch in range(EPOCHS):
        model.train()
        criterion.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model(batch_x)
            loss = criterion(embeddings, batch_y)
            loss.backward()
            criterion.constrained_weight_update()
            optimizer.step()
            total_loss += loss.item()

        # Record prototypes
        tracker.record(criterion.prototypes, epoch=epoch+1)

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_embs, test_targs = [], []
            for bx, by in test_loader:
                test_embs.append(model(bx.to(DEVICE)))
                test_targs.append(by.to(DEVICE))
            test_embs = torch.cat(test_embs)
            test_targs = torch.cat(test_targs)

            recalls = compute_recall_at_k(test_embs, test_targs, k_values=(1, 2, 4, 8))
            stats = criterion.get_last_stats()

        history['epoch'].append(epoch+1)
        history['loss'].append(total_loss/len(train_loader))
        history['r@1'].append(recalls['R@1'])
        history['r@2'].append(recalls['R@2'])

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"R@1: {recalls['R@1']:.4f} | R@2: {recalls['R@2']:.4f} | "
              f"Proto Mvmt: {stats.get('proto_movement_mean', 0):.4f}")

    print("\n✓ Training complete!\n")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    print("="*70)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    final_prototypes = criterion.prototypes.detach().cpu()

    # 1. Initial state visualization
    print("1️⃣  Visualizing prototype initialization...")
    fig1 = plot_prototype_initialization(
        initial_prototypes.cpu(),
        initial_kmeans=None,
        X_train=X_train,
        y_train=y_train,
        n_classes=N_CLASSES
    )
    if fig1:
        plt.savefig('01_initialization.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: 01_initialization.png\n")

    # 2. Prototype movement
    print("2️⃣  Visualizing prototype movement (Random → Trained)...")
    fig2 = plot_prototype_movement(
        initial_prototypes.cpu(),
        final_prototypes,
        X_train=X_train,
        y_train=y_train,
        n_classes=N_CLASSES
    )
    if fig2:
        plt.savefig('02_movement.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: 02_movement.png\n")

    # 3. Evolution trajectory
    print("3️⃣  Visualizing prototype evolution over epochs...")
    fig3 = plot_prototype_trajectory(
        tracker,
        initial_prototypes.cpu(),
        X_train=X_train,
        y_train=y_train,
        n_classes=N_CLASSES,
        sample_epochs=5
    )
    if fig3:
        plt.savefig('03_trajectory.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: 03_trajectory.png\n")

    # 4. Movement distance over time
    print("4️⃣  Visualizing movement distance metric...")
    fig4 = plot_movement_distance(tracker, initial_prototypes.cpu())
    if fig4:
        plt.savefig('04_distance.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: 04_distance.png\n")

    # 5. Training metrics
    print("5️⃣  Plotting training metrics...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['epoch'], history['loss'], 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['r@1'], 'o-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Recall@1')
    axes[1].set_title('Recall@1'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['epoch'], history['r@2'], 'o-', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Recall@2')
    axes[2].set_title('Recall@2'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_training_metrics.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: 05_training_metrics.png\n")

    print("="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\n📊 Generated files:")
    print("   • 01_initialization.png - Initial random prototypes")
    print("   • 02_movement.png      - Prototype movement arrows")
    print("   • 03_trajectory.png    - Evolution over epochs")
    print("   • 04_distance.png      - L2 distance metric")
    print("   • 05_training_metrics.png - Loss & Recall@K\n")