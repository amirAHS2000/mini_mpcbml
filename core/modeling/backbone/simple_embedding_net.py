import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """
    For 2D setting: Apply only diagonal scaling (element-wise multiplication)
    followed by L2 normalization.
    
    This learns a scale factor per dimension:
    output = normalize(input * diag_scale)
    """
    def __init__(self, input_dim=2, hidden_dim=None, output_dim=2):
        super().__init__()
        # Ensure input_dim == output_dim for diagonal scaling
        assert input_dim == output_dim, \
            f"For diagonal scaling: input_dim ({input_dim}) must equal output_dim ({output_dim})"
        
        # Learnable diagonal scaling [output_dim]
        self.scale = nn.Parameter(torch.ones(output_dim))
    
    def forward(self, x):
        # x: [B, D]
        # Apply diagonal scaling: element-wise multiply
        scaled = x * self.scale
        # Normalize to unit norm
        return F.normalize(scaled, p=2, dim=1)
