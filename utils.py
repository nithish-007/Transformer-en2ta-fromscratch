import torch 
from torch import nn
import math

# ---------------------------------
# Input Embeddings
# ---------------------------------

class InputEmbeddings(nn.Module):
    """
    Converts input token indices into dense vector representations
    """
    def __init__(self, d_model:int, vocab_size:int) -> None:
        '''
        Args:
            d_model (int): Dimension of each embedding vector
            vocab_size (int): size/ total number of unique tokens in the vocabulary
        '''
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Tensor of shape (batch, seq_len) containing token indices
        Returns:
            embeddings (torch.Tensor): Tensor of shape (batch, seq_len, d_model) containing the embedded representations
        '''
        # Multiply by sqrt(d_model) to scale the embeddings according to the the paper
        return self.embeddings(x) * math.sqrt(self.d_model)


# ---------------------------------
# Positional Embeddings
# ---------------------------------
# Sinusoidal Positional Encoding

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        '''
        Args:
            d_model (int): Dimension of each embedding vector
            seq_len (int): number of positions (max sequence length)
            dropout (float): dropout rate
        '''
        # pe[:, 0::2] = sin(pos / 10000**(2i/d_model))
        # pe[:, 1::2] = cos(pos / 10000**(2i/d_model))
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a pos_enc matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # position vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # div term in the equation of shape (d_model, )
        # using log-term instead of direct formula to get a numerical stability (avoid overflow and underflow)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #(d_model / 2)
        
        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) 
        # apply cosine to off indices 
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding 
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer (to avoid it as a learnablr parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        # And in this above, slicing the Seq_len only upto the seq_len of i/p
        return self.dropout(x)
    

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


# ----------------------
# FeedForwardBlock
# ----------------------

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)         # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model)   # Second linear transformation

    def forward(self, x):
        # Two linear transformations with ReLU activation and dropout in between
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# ---------------------
# ResidualConnection
# ---------------------

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features) # features --> d_model (we do norm across dim(512) for each word)
    
    def forward(self, x, sublayer):
        # Appling normalization, then the sublayer, then dropout, and add to input (skip connection)
        return x + self.dropout(sublayer(self.norm(x)))
  
# -------------------------------------------------
# ProjectionLayer(nn.Module) - to get probability distro at decoder end
# --------------------------------------------------    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size) # Linear layer to vocab size

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection(x)
   
