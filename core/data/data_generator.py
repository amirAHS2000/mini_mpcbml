import numpy as np

class SyntheticGaussianMixture:
    """
    Generate synthetic 2D Gaussian clusters with class split for few-shot evaluation.
    
    Example:
        Total 20 classes arranged on a circle.
        Train on classes 0-14 (15 classes).
        Test on classes 15-19 (5 unseen classes).
    """
    
    def __init__(self, n_classes_total=20, n_classes_train=15,
                 modes_per_class=3, n_samples=3000,
                 noise_std=0.10, radius=2.0):
        """
        Args:
            n_classes_total: Total number of classes (e.g., 20 for CUB-like)
            n_classes_train: Number of classes for training (e.g., 15)
            modes_per_class: Prototypes per class
            n_samples: Total samples
            noise_std: Gaussian noise for modes
            radius: Distance of clusters from origin
        """
        self.n_classes_total = n_classes_total
        self.n_classes_train = n_classes_train
        self.n_classes_test = n_classes_total - n_classes_train
        self.modes_per_class = modes_per_class
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.radius = radius
        
        assert n_classes_train < n_classes_total, \
            "Train classes must be less than total classes"
    
    def generate(self, return_split=True):
        """
        Generate data with optional train/test split.
        
        Args:
            return_split: If True, return (train_data, test_data).
                         If False, return all data (X, y).
        
        Returns:
            If return_split:
                (X_train, y_train, X_test, y_test)
            Else:
                (X, y)
        """
        total_modes = self.n_classes_total * self.modes_per_class
        X_list, y_list = [], []
        angles = np.linspace(0, 2 * np.pi, total_modes, endpoint=False)
        sample_per_mode = self.n_samples // total_modes
        
        for i, angle in enumerate(angles):
            class_label = i % self.n_classes_total
            centroid = np.array([
                self.radius * np.cos(angle),
                self.radius * np.sin(angle)
            ])
            blob_X = np.random.randn(sample_per_mode, 2) * self.noise_std + centroid
            X_list.append(blob_X)
            y_list.append(np.full(sample_per_mode, class_label))
        
        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.int64)
        
        if return_split:
            # Split by class: train classes 0...n_classes_train-1
            #                test classes n_classes_train...n_classes_total-1
            train_mask = y < self.n_classes_train
            test_mask = y >= self.n_classes_train
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            # Relabel test classes to 0...n_classes_test-1 for clarity
            y_test = y_test - self.n_classes_train
            
            return X_train, y_train, X_test, y_test
        else:
            return X, y
