import torch 
from torch import nn

# --------------------------
# LayerNormalization
# --------------------------

class LayerNormalization(nn.Module):
    '''
    Normalize the data across dim
    '''
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps # small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)

        # Normalize and apply learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


