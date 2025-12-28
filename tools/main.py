# main_updated.py - USING NEW PLUGINS (Visualization Logger + Prototype Initializer Factory)
# Shows how to integrate the new modular systems

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

from utils.config import load_config

from data.base_dataset import SimpleDataset
from data.data_generator import SyntheticGaussianMixture
from modeling.backbone.simple_embedding_net import EmbeddingNet
from losses.mpcbml_loss import MpcbmlLoss
from data.evaluation.retrieval_metric import compute_recall_at_k
from utils.prototype_tracker import PrototypeTracker
from utils.prototype_logger import plot_prototype_initialization
from solver.build_optimizer import build_optimizer, build_lr_scheduler, print_optimizer_config
from utils.visualization_logger import VisualizationLogger, save_prototypes_visualization
from utils.prototype_initializer_factory import PrototypeInitializerFactory


def print_banner(text: str):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(f"🚀 {text}")
    print("="*70 + "\n")


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
    # 🆕 VISUALIZATION LOGGER SETUP
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
    
    print("Generating synthetic dataset...")
    gen = SyntheticGaussianMixture(
        n_classes=N_CLASSES,
        modes_per_class=K_PROTOS,
        n_samples=cfg.DATA.N_SAMPLES
    )
    X, y = gen.generate()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.DATA.TEST_SPLIT,
        stratify=y
    )
    
    train_loader = DataLoader(
        SimpleDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATA.NUM_WORKERS
    )
    test_loader = DataLoader(
        SimpleDataset(X_test, y_test),
        batch_size=cfg.DATA.TEST_BATCHSIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS
    )
    
    print(f"✓ Dataset created: {len(X_train)} training, {len(X_test)} test\n")
    
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
    # 🆕 PROTOTYPE INITIALIZATION WITH FACTORY PATTERN
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
    
    print(f"✓ Prototypes initialized with '{init_method}' method")
    print(f"✓ Shape: {initial_prototypes.shape}\n")
    
    # Create prototype tracker
    tracker = PrototypeTracker(N_CLASSES, K_PROTOS, EMBED_DIM)
    tracker.record(criterion.prototypes, epoch=0)
    
    # ====================================================================
    # OPTIMIZER & SCHEDULER SETUP
    # ====================================================================
    print_banner("OPTIMIZER & SCHEDULER SETUP")
    
    optimizer_main, optimizer_weights = build_optimizer(cfg, model, criterion)
    scheduler_main, scheduler_weights = build_lr_scheduler(cfg, optimizer_main, optimizer_weights)
    
    print_optimizer_config(optimizer_main, optimizer_weights)
    
    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    print_banner("STARTING TRAINING")
    
    history = {
        'epoch': [],
        'loss': [],
        'r@1': [],
        'r@2': [],
        'r@4': [],
        'r@8': []
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
        
        scheduler_main.step()
        if scheduler_weights is not None:
            scheduler_weights.step()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_embs, test_targs = [], []
            for bx, by in test_loader:
                test_embs.append(model(bx.to(DEVICE)))
                test_targs.append(by.to(DEVICE))
            
            test_embs = torch.cat(test_embs)
            test_targs = torch.cat(test_targs)
            
            recalls = compute_recall_at_k(
                test_embs,
                test_targs,
                k_values=tuple(cfg.VALIDATION.RECALL_K)
            )
            stats = criterion.get_last_stats()
        
        # Store metrics
        history['epoch'].append(epoch+1)
        history['loss'].append(total_loss / len(train_loader))
        history['r@1'].append(recalls['R@1'])
        history['r@2'].append(recalls['R@2'])
        history['r@4'].append(recalls['R@4'])
        history['r@8'].append(recalls['R@8'])
        
        # Print progress
        if (epoch + 1) % cfg.VALIDATION.VERBOSE == 0:
            proto_mvmt = stats.get('proto_movement_mean', 0) if stats else 0
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"R@1: {recalls['R@1']:.4f}")
    
    print("\n✓ Training complete!\n")
    
    # ====================================================================
    # 🆕 VISUALIZATIONS USING SEPARATED LOGGER
    # ====================================================================
    print_banner("GENERATING VISUALIZATIONS")
    
    final_prototypes = criterion.prototypes.detach().cpu()
    
    # Save prototype visualizations (if enabled in config)
    viz_config = getattr(cfg, 'VISUALIZATION', None)
    if viz_config:
        save_prototypes_visualization(
            output_dir=output_dir,
            initial_prototypes=initial_prototypes,
            final_prototypes=final_prototypes,
            X_train=X_train,
            y_train=y_train,
            n_classes=N_CLASSES,
            tracker=tracker,
            save_initialization=getattr(viz_config, 'SAVE_INITIALIZATION', True),
            save_movement=getattr(viz_config, 'SAVE_MOVEMENT', True),
            save_trajectory=getattr(viz_config, 'SAVE_TRAJECTORY', True),
            save_distance=getattr(viz_config, 'SAVE_DISTANCE', True)
        )
    
    # Save training metrics using logger
    print("\n5️⃣  Plotting training metrics...")
    viz_logger.plot_training_metrics(history, '05_metrics.png')
    
    # ====================================================================
    # 🆕 SAVE RESULTS USING LOGGER
    # ====================================================================
    print_banner("SAVING RESULTS")
    
    viz_logger.save_summary(history, args.config, 'summary.txt')
    viz_logger.print_results(history)


if __name__ == '__main__':
    main()
