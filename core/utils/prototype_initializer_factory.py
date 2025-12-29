import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Callable, Optional


class PrototypeInitializerFactory:
    """
    Factory for prototype initialization methods.
    Allows plugging different initialization strategies via config.
    """
    
    def __init__(self):
        """Initialize the factory with available methods."""
        self.methods = {
            'random': self.initialize_random,
            'kmeans': self.initialize_kmeans,
            'mean': self.initialize_mean,
        }
    
    @staticmethod
    def initialize_random(
        n_classes: int,
        prototype_per_class: int,
        embed_dim: int,
        device: str,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random initialization.
        
        Args:
            n_classes: Number of classes
            prototype_per_class: Prototypes per class
            embed_dim: Embedding dimension
            device: Device (cuda/cpu)
            **kwargs: Unused extra arguments
        
        Returns:
            Tuple of (prototypes, cluster_sizes)
        """
        prototypes = torch.randn(
            n_classes,
            prototype_per_class,
            embed_dim,
            device=device
        )
        
        # Equal cluster sizes for random init
        cluster_sizes = torch.ones(
            n_classes,
            prototype_per_class,
            device=device
        ) / prototype_per_class
        
        return prototypes, cluster_sizes
    
    @staticmethod
    def initialize_kmeans(
        model: torch.nn.Module,
        train_loader,
        n_classes: int,
        prototype_per_class: int,
        embed_dim: int,
        device: str,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        K-means based initialization.
        
        Args:
            model: Model to extract embeddings
            train_loader: Training data loader
            n_classes: Number of classes
            prototype_per_class: Prototypes per class
            embed_dim: Embedding dimension
            device: Device (cuda/cpu)
            **kwargs: Unused extra arguments
        
        Returns:
            Tuple of (prototypes, cluster_sizes)
        """
        model.eval()
        
        # Collect features per class
        class_features = {cls: [] for cls in range(n_classes)}
        
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                embeddings = model(batch_x)
                
                # Group by class
                for i, cls_idx in enumerate(batch_y):
                    class_features[int(cls_idx)].append(embeddings[i].cpu().numpy())
        
        prototypes = torch.zeros(
            n_classes,
            prototype_per_class,
            embed_dim,
            device=device
        )
        
        cluster_sizes = torch.zeros(
            n_classes,
            prototype_per_class,
            device=device
        )
        
        # K-means for each class
        for cls_idx in range(n_classes):
            if len(class_features[cls_idx]) == 0:
                # Random init if no samples for this class
                prototypes[cls_idx] = torch.randn(
                    prototype_per_class,
                    embed_dim,
                    device=device
                )
                cluster_sizes[cls_idx] = torch.ones(
                    prototype_per_class,
                    device=device
                ) / prototype_per_class
                continue
            
            # Stack features
            feats_np = np.array(class_features[cls_idx])
            
            if len(feats_np) >= prototype_per_class:
                # K-means clustering
                kmeans = KMeans(
                    n_clusters=prototype_per_class,
                    random_state=0,
                    n_init=10
                ).fit(feats_np)
                
                centers = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=torch.float,
                    device=device
                )
                prototypes[cls_idx] = centers
                
                # Cluster sizes (normalized)
                cluster_counts = np.bincount(
                    kmeans.labels_,
                    minlength=prototype_per_class
                )
                cluster_sizes[cls_idx] = torch.tensor(
                    cluster_counts / len(kmeans.labels_),
                    dtype=torch.float,
                    device=device
                )
            else:
                # Mean init if not enough samples
                mean_feat = torch.tensor(
                    feats_np.mean(axis=0),
                    dtype=torch.float,
                    device=device
                )
                
                for k in range(prototype_per_class):
                    noise = torch.randn(embed_dim, device=device) * 0.01
                    prototypes[cls_idx, k] = mean_feat + noise
                
                cluster_sizes[cls_idx] = torch.ones(
                    prototype_per_class,
                    device=device
                ) / prototype_per_class
        
        return prototypes, cluster_sizes
    
    @staticmethod
    def initialize_mean(
        model: torch.nn.Module,
        train_loader,
        n_classes: int,
        prototype_per_class: int,
        embed_dim: int,
        device: str,
        noise_scale: float = 0.01,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean-based initialization with noise perturbation.
        
        Args:
            model: Model to extract embeddings
            train_loader: Training data loader
            n_classes: Number of classes
            prototype_per_class: Prototypes per class
            embed_dim: Embedding dimension
            device: Device (cuda/cpu)
            noise_scale: Scale of noise perturbation
            **kwargs: Unused extra arguments (can include 'noise_scale')
        
        Returns:
            Tuple of (prototypes, cluster_sizes)
        """
        # Allow noise_scale to be passed in kwargs
        if 'noise_scale' in kwargs:
            noise_scale = kwargs['noise_scale']
        
        model.eval()
        
        # Collect features per class
        class_features = {cls: [] for cls in range(n_classes)}
        
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                embeddings = model(batch_x)
                
                # Group by class
                for i, cls_idx in enumerate(batch_y):
                    class_features[int(cls_idx)].append(embeddings[i].cpu().numpy())
        
        prototypes = torch.zeros(
            n_classes,
            prototype_per_class,
            embed_dim,
            device=device
        )
        
        cluster_sizes = torch.ones(
            n_classes,
            prototype_per_class,
            device=device
        ) / prototype_per_class
        
        # Mean + noise for each class
        for cls_idx in range(n_classes):
            if len(class_features[cls_idx]) == 0:
                # Random init if no samples
                prototypes[cls_idx] = torch.randn(
                    prototype_per_class,
                    embed_dim,
                    device=device
                )
                continue
            
            # Compute mean
            feats_np = np.array(class_features[cls_idx])
            class_mean = torch.tensor(
                feats_np.mean(axis=0),
                dtype=torch.float,
                device=device
            )
            
            # Add noise perturbations
            for k in range(prototype_per_class):
                noise = torch.randn(embed_dim, device=device) * noise_scale
                prototypes[cls_idx, k] = class_mean + noise
        
        return prototypes, cluster_sizes
    
    def get_initializer(self, method_name: str) -> Optional[Callable]:
        """
        Get initializer function by name.
        
        Args:
            method_name: Name of initialization method
        
        Returns:
            Initializer function or None if not found
        """
        return self.methods.get(method_name.lower())
    
    def initialize(
        self,
        method: str,
        model: Optional[torch.nn.Module] = None,
        train_loader=None,
        n_classes: int = 4,
        prototype_per_class: int = 3,
        embed_dim: int = 64,
        device: str = 'cpu',
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize prototypes using specified method.
        
        Args:
            method: Name of initialization method ('random', 'kmeans', 'mean')
            model: Model for feature extraction (required for kmeans/mean)
            train_loader: Training data loader (required for kmeans/mean)
            n_classes: Number of classes
            prototype_per_class: Prototypes per class
            embed_dim: Embedding dimension
            device: Device (cuda/cpu)
            **kwargs: Extra arguments for specific methods
        
        Returns:
            Tuple of (prototypes, cluster_sizes)
        
        Raises:
            ValueError: If method not found or invalid arguments
        """
        method_lower = method.lower()
        
        if method_lower not in self.methods:
            available = ', '.join(self.methods.keys())
            raise ValueError(
                f"Unknown initialization method: {method}. "
                f"Available methods: {available}"
            )
        
        initializer = self.methods[method_lower]
        
        # Random init doesn't need model/loader
        if method_lower == 'random':
            return initializer(
                n_classes,
                prototype_per_class,
                embed_dim,
                device,
                **kwargs
            )
        
        # Other methods need model and loader
        if model is None or train_loader is None:
            raise ValueError(
                f"Method '{method}' requires 'model' and 'train_loader' arguments"
            )
        
        return initializer(
            model,
            train_loader,
            n_classes,
            prototype_per_class,
            embed_dim,
            device,
            **kwargs
        )
    
    def list_methods(self) -> list:
        """List available initialization methods."""
        return list(self.methods.keys())