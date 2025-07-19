import torch
from torch import nn
import math

# ---------------------------------
# Input Embeddings
# ---------------------------------

class InputEnbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forwad(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the the paper
        return self.embeddings(x) * math.sqrt(self.d_model)

# ---------------------------------
# Positional Embeddings
# ---------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #(d_model / 2)
        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position / 10000**(2i / d_model))
        # apply cosine to off indices 
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position / 10000**(2i / d_model))
        # Add a batch dimension to the positional encoding 
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
# -----------------------------------
# Multi-head Attention
# -----------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads

        assert self.d_model % self.h == 0 # d_model is not divisible by h

       
    