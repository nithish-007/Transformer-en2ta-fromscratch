import torch
from torch import nn
from utils import MultiHeadAttention, FeedForwardBlock, LayerNormalization, ResidualConnection


# ------------------------------
# EncoderBlock (single encoder layer)
# ------------------------------
class EncoderBlock(nn.Module):
    ''' 
    Single encoder layer consisting of self-attention and feed-forward network with residual connections and layer normalization.
    '''
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        ''' 
        Args:
            d_model (int): Dimension of each embedding vector
            self_attention_block (MultiHeadAttention): Multi-head self-attention block
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
            dropout (float): Dropout rate for regularization
        '''
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        ''' 
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            src_mask (torch.Tensor): Source mask for attention mechanism
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch, seq_len, d_model)
        '''
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

     
# ------------------------------
# Encoder (full encoder network)
# ------------------------------
class Encoder(nn.Module):
    ''' 
    Full encoder network consisting of multiple encoder layers followed by layer normalization.
    '''
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        ''' 
        Args:
            d_model (int): Dimension of each embedding vector
            layers (nn.ModuleList): List of EncoderBlock layers
        '''
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model) #d_model --> features

    def forward(self, x, mask):
        '''
        Args: 
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            mask (torch.Tensor): Source mask for attention mechanism
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch, seq_len, d_model)
        '''
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    