import torch 
from torch import nn
from src.utils import (MultiHeadAttention, FeedForwardBlock, LayerNormalization, ResidualConnection,
                   )
# -------------------------
# DecoderBlock (single decoder layer)
# -------------------------
class DecoderBlock(nn.Module):
    ''' 
    Single decoder layer consisting of self-attention, cross-attention, and feed-forward network with residual connections and layer normalization.
    '''
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, 
                 cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        ''' 
        Args:
            d_model (int): Dimension of each embedding vector
            self_attention_block (MultiHeadAttention): Multi-head self-attention block
            cross_attention_block (MultiHeadAttention): Multi-head cross-attention block
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward network
            dropout (float): Dropout rate for regularization
        '''
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        ''' 
        Args:
            x (torch.Tensor): Target input tensor of shape (batch, seq_len, d_model)
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch, seq_len, d_model)
            src_mask (torch.Tensor): Source mask for attention mechanism
            tgt_mask (torch.Tensor): Target mask for attention mechanism

        Returns:
            x (torch.Tensor): Target Output tensor of shape (batch, seq_len, d_model)
        '''
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x   
    
# ------------------------------
# Decoder (full decoder network)
# ------------------------------

class Decoder(nn.Module):
    '''
    Full decoder network consisting of multiple decoder layers followed by layer normalization.
    '''
    def __init__(self, d_model: int, layers: nn.ModuleList):
        '''
        Args:
            d_model (int): Dimension of each embedding vector
            layers (nn.ModuleList): List of DecoderBlock layers

        '''
        super().__init__()  
        self.layers = layers
        self.norm = LayerNormalization(d_model) # d_model --> features

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        ''' 
        Args:
            x (torch.Tensor): Target input tensor of shape (batch, seq_len, d_model)
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch, seq_len, d_model)
            src_mask (torch.Tensor): Source mask for attention mechanism
            tgt_mask (torch.Tensor): Target mask for attention mechanism
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch, seq_len, d_model)
        '''
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
    
# -------------------------------------------------
# ProjectionLayer(nn.Module) - to get probability distro at decoder end
# --------------------------------------------------    

class ProjectionLayer(nn.Module):
    ''' 
    Project the decoder output to the vocabulary size to get the logits for each token
    1. Linear Transformation: Applies a linear transformation to map the decoder output to the vocabulary size
    2. Output Shape: The output shape will be (batch, seq_len, vocab_size)
    3. This layer is typically followed by a softmax function during training to obtain the probability distribution over the vocabulary
    4. This layer shares weights with the input embedding layer in some implementations to reduce the number of parameters
    '''
    def __init__(self, d_model, vocab_size) -> None:
        ''' 
        Args:
            d_model (int): Dimension of each embedding vector
            vocab_size (int): size/total number of unique tokens(words) in the vocabulary
        '''
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        ''' 
        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch, seq_len, d_model)
        Returns:
            torch.Tensor: Resultant matrix of shape (batch, seq_len, vocab_size) containing the logits for each token in the vocabulary
        '''
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection(x)