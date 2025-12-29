class PrototypeTracker:
    """Tracks prototype movement throughout training."""

    def __init__(self, num_classes, prototype_per_class, embed_dim):
        self.num_classes = num_classes
        self.prototype_per_class = prototype_per_class
        self.embed_dim = embed_dim
        self.history = []
        self.epoch_numbers = []

    def record(self, prototypes, epoch):
        """Store prototypes at current epoch."""
        self.history.append(prototypes.detach().cpu().clone())
        self.epoch_numbers.append(epoch)

    def get_movement(self, initial_protos):
        """Compute L2 distance from initial prototypes at each epoch."""
        movements = []
        initial_flat = initial_protos.reshape(self.num_classes * self.prototype_per_class, -1)

        for proto_state in self.history:
            proto_flat = proto_state.reshape(self.num_classes * self.prototype_per_class, -1)
            dist = (proto_flat - initial_flat).norm(dim=1).mean().item()
            movements.append(dist)

        return movements