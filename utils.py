import stat
import torch
from torch import nn
import math

# ---------------------------------
# Embedding Layer
# ---------------------------------

class EmbeddingLayer(nn.Module):
    """
    Converts input token indices into dense vector representations
    """
    def __init__(self, vocab_size:int, d_model:int) -> None:
        '''
        Args:
            d_model (int): Dimension of each embedding vector
            vocab_size (int): size/ total number of unique tokens(words) in the vocabulary
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
    ''' 
    Add sinusoidal positional encoding to the input embeddings
    1. pe(pos, 2i) = sin(pos / 10000^(2i/d_model)) --> pe[:, 0::2] = sin(pos / 10000**(2i/d_model))
    2. pe(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) --> pe[:, 1::2] = cos(pos / 10000**(2i/d_model))
    where pos is the position and i is the dimension
    '''
    
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        '''
        Args:
            d_model (int): Dimension of each embedding vector
            seq_len (int): number of positions (max sequence length) // length of the input sequence
            dropout (float): dropout rate
        '''
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
        # Register the positional encoding as a buffer (to avoid it as a learnable parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        ''' 
        Args:
            x (torch.Tensor): Tensor of shape (batch, seq_len, d_model) containing input embeddings
        Returns:
            x (torch.Tensor): Tensor of shape (batch, seq_len, d_model) containing input embeddings with positional encoding added
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        # And in this above, slicing the Seq_len only upto the seq_len of i/p
        return self.dropout(x)
 

# --------------------------
# LayerNormalization
# --------------------------

class LayerNormalization(nn.Module):
    '''
    Normalize the data across dimensions for each data point(words) independently
    1. mean = 1/d * sum(x_i) for i in dimensions
    2. std = sqrt(1/d * sum((x_i - mean)^2)) for i in dimensions
    3. output = alpha * (x - mean) / (std + eps) + bias
    where alpha and bias are learnable parameters
    '''
    def __init__(self, d_model: int) -> None:
        ''' 
        Args:
            d_model (int): number of features/ dimensions in the input
        '''
        super().__init__()
        self.eps = 1e-5 # small value to avoid division by zero
        self.features = d_model # number of features/ dimensions in the input
        self.alpha = nn.Parameter(torch.ones(self.features))
        self.bias = nn.Parameter(torch.zeros(self.features))

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Tensor of shape (batch, seq_len, d_model) containing input data
        Returns:
            x (torch.Tensor): Tensor of shape (batch, seq_len, d_model) containing normalized data
        ''' 
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)

        # Normalize and apply learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# ---------------------
# ResidualConnection
# ---------------------

class ResidualConnection(nn.Module):
    ''' 
    Implements a residual connection followed by layer normalization and dropout
    '''
    def __init__(self, d_model: int, dropout: float) -> None:
        ''' 
        Args:
            d_model (int): dimension of each embedding vector (features)
            dropout (float): dropout rate
        '''
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model) # d_model (we do norm across dim(512) for each word)
    
    def forward(self, x, sublayer):
        ''' 
        Args:
            x (torch.Tensor): input of shape (batch, seq_len, d_model)
            sublayer (nn.Module): the sublayer to be applied after normalization --> could be attention or feed-forward network
        
        Returns:
            x (torch.Tensor): output of shape (batch, seq_len, d_model)
        '''
        # Appling normalization, then the sublayer, then dropout, and add to input (skip connection)
        # note: paper uses post norm but here we use pre-norm for better training stability
        return x + self.dropout(sublayer(self.norm(x)))

 
# ----------------------
# FeedForwardBlock
# ----------------------

class FeedForwardBlock(nn.Module):
    ''' 
    Implements the position-wise feed-forward network 
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        ''' 
        Args:
            d_model (int): dimension of each embedding vector
            d_ff (int): dimension of the feed-forward network
            dropout (float): dropout rate
        '''
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)         # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model)   # Second linear transformation

    def forward(self, x):
        ''' 
        Args:
            x (torch.Tensor): input of shape (batch, seq_len, d_model)
        Returns:
            x (torch.Tensor): output of shape (batch, seq_len, d_model) 
        '''
        # Two linear transformations with ReLU activation and dropout in between
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# -----------------------------------
# Multi-head Attention
# -----------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads

        assert self.d_model % self.h == 0 # d_model should be divisible by h

        self.d_k = self.d_model // self.h # dim for each heads
        # Linear transformations for queries, keys, and values
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores = torch.masked_fill(attention_scores, mask==0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # replicating input embeddings into query, key, value using linear transformation
        query = self.w_q(q) # input(h_i's)(batch, seq_len, d_model) --> query(batch, seq_len, d_model)
        key = self.w_k(k) # input(h_i's)(batch, seq_len, d_model) --> key(batch, seq_len, d_model)
        value = self.w_v(v) # input(h_i's)(batch, seq_len, d_model) --> key(batch, seq_len, d_model)

        # split the query, key, value into multiple heads 
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_model)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Calucate the attention
        x, self.attention_scores = self.scaled_dot_product(query, key, value, mask, self.dropout)

        # concatanate the heads 
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # Matrix Transformation (W_o @ x)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    

def causal_mask(seq_len):
    """
    Causal mask ensures that positions above the main diagonal are masked out, preventing the model from attending to future tokens during decoding.

    Args:
        seq_len (int): Length of the sequence to generate the mask for.

    Returns:
        torch.BoolTensor: A mask tensor of shape (1, seq_len, seq_len) where True values indicate allowed positions and False values indicate masked positions.
    """
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)    # All the values above diagonal will be 1
    return mask == 0    # All the values above diagonal will be 0, and rest will be 1