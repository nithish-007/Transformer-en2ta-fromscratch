import torch 
from torch import nn
from utils import ResidualConnection, LayerNormalization, FeedForwardBlock
from attention import MultiHeadAttention

# ------------------------------
# EncoderBlock (single encoder layer)
# ------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

    
# ------------------------------
# Encoder (full encoder network)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
