import numpy as np

class SyntheticGaussianMixture:
    def __init__(self, n_classes=4, modes_per_class=3, n_samples=2400,
                 noise_std=0.10, radius=2.0):
        self.n_classes = n_classes
        self.modes_per_class = modes_per_class
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.radius = radius

    def generate(self):
        total_modes = self.n_classes * self.modes_per_class
        X_list, y_list = [], []
        angles = np.linspace(0, 2 * np.pi, total_modes, endpoint=False)
        sample_per_mode = self.n_samples // total_modes

        for i, angle in enumerate(angles):
            class_label = i % self.n_classes
            centroid = np.array([self.radius * np.cos(angle), self.radius * np.sin(angle)])
            blob_X = np.random.randn(sample_per_mode, 2) * self.noise_std + centroid
            X_list.append(blob_X)
            y_list.append(np.full(sample_per_mode, class_label))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X.astype(np.float32), y.astype(np.int64)