import torch
import torch.optim as optim
from types import SimpleNamespace


def build_optimizer(cfg, model, criterion=None):
    """
    Build optimizers for MP-CBML training.
    
    Separates network and loss parameters across different optimizers:
    - optimizer_main (Adam): Model parameters + prototypes + temperature
    - optimizer_weights (SGD): Mixture weights ONLY (for constraint preservation)
    
    Args:
        cfg: Configuration object with learning rate settings
        model: Neural network model
        criterion: Loss criterion (MpcbmlLoss for MP-CBML)
    
    Returns:
        optimizer_main: Adam optimizer for network and most loss parameters
        optimizer_weights: SGD optimizer for mixture weights (MP-CBML only), or None
    """
    
    params = []
    base_lr = cfg.get('base_lr', 0.001)
    
    # ================================================================
    # Add model parameters to Adam
    # ================================================================
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Standard learning rate for all model parameters
        lr_mul = 1.0
        
        params.append({
            'params': [param],
            'lr': base_lr * lr_mul
        })
    
    # ================================================================
    # Add criterion parameters to Adam (except weights for MP-CBML)
    # ================================================================
    optimizer_weights = None
    
    if criterion is not None:
        # Check if this is MP-CBML loss
        is_mpcbml = hasattr(criterion, 'weights') and hasattr(criterion, 'prototypes')
        
        for name, param in criterion.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip mixture weights - they get their own SGD optimizer
            if name == 'weights' and is_mpcbml:
                continue
            
            # Determine learning rate multiplier for each parameter type
            if 'prototypes' in name:
                # Prototypes need to move fast to track data clusters
                # Using 100x LR matches literature (ProxyNCA, Proxy-Anchor)
                current_lr_mul = 100.0
            elif 'theta' in name:
                # Temperature parameter - standard learning rate
                current_lr_mul = 1.0
            else:
                # Other parameters (class priors, etc.)
                current_lr_mul = 1.0
            
            params.append({
                'params': [param],
                'lr': base_lr * current_lr_mul
            })
        
        # ================================================================
        # Build separate SGD optimizer for mixture weights (MP-CBML only)
        # ================================================================
        if is_mpcbml and hasattr(criterion, 'weights'):
            # SGD with separate learning rate configuration
            weight_lr = cfg.get('weight_lr', 0.0001)
            weight_momentum = cfg.get('weight_momentum', 0.9)
            
            optimizer_weights = optim.SGD(
                [{"params": [criterion.weights]}],
                lr=weight_lr,
                momentum=weight_momentum,
                weight_decay=0.0  # NO weight decay on weights (would violate constraint)
            )
    
    # ================================================================
    # Build main optimizer (Adam) for network + most loss params
    # ================================================================
    weight_decay = cfg.get('weight_decay', 0.0)
    
    optimizer_main = optim.Adam(
        params,
        weight_decay=weight_decay
    )
    
    return optimizer_main, optimizer_weights


def build_lr_scheduler(cfg, optimizer_main, optimizer_weights=None):
    """
    Build learning rate schedulers for both optimizers.
    
    Args:
        cfg: Configuration object with scheduler settings
        optimizer_main: Main optimizer (Adam)
        optimizer_weights: Weights optimizer (SGD, optional)
    
    Returns:
        scheduler_main: Learning rate scheduler for main optimizer
        scheduler_weights: Learning rate scheduler for weights optimizer (or None)
    """
    
    # ================================================================
    # Scheduler for main optimizer (network + prototypes + temperature)
    # ================================================================
    scheduler_type = cfg.get('scheduler_type', 'cosine')
    epochs = cfg.get('epochs', 20)
    
    if scheduler_type == 'cosine':
        eta_min = cfg.get('eta_min_ratio', 0.01)
        scheduler_main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main,
            T_max=epochs,
            eta_min=optimizer_main.defaults['lr'] * eta_min
        )
    else:
        # Fallback to step scheduler
        scheduler_main = optim.lr_scheduler.StepLR(
            optimizer_main,
            step_size=epochs // 3,
            gamma=0.1
        )
    
    # ================================================================
    # Scheduler for weights optimizer (if exists)
    # ================================================================
    scheduler_weights = None
    if optimizer_weights is not None:
        if scheduler_type == 'cosine':
            eta_min = cfg.get('eta_min_ratio', 0.01)
            scheduler_weights = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_weights,
                T_max=epochs,
                eta_min=optimizer_weights.defaults['lr'] * eta_min
            )
        else:
            scheduler_weights = optim.lr_scheduler.StepLR(
                optimizer_weights,
                step_size=epochs // 3,
                gamma=0.1
            )
    
    return scheduler_main, scheduler_weights
