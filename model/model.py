import torch
from torch import nn
import math

# ---------------------------------
# Input Embeddings
# ---------------------------------

class InputEnbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
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
        '''
        pe[:, 0::2] = sin(pos / 10000**(2i/d_model))
        pe[:, 1::2] = cos(pos / 10000**(2i/d_model))
        '''
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

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
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # (batch, seq_len, d_model)
        return self.dropo2ut(x)
    
# -----------------------------------
# Multi-head Attention
# -----------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads

        assert self.d_model % self.h == 0 # d_model is divisible by h

        self.d_k = self.d_model // self.h # dim for each heads
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
        attention_scores = torch.softmax(attention_scores, dim=1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # replicating inout embeddings into query, key, value using linear transformation
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

# --------------------------------
# LayerNormalization
# --------------------------------

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float = 10**-6) -> None:
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim = True) # --> (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # --> (batch, seq_len , 1)

        # eps is to prevent dividing by zero or when std is very small 
        return self.alpha * (x - mean) / (std + self.eps) + __build_class__

# ---------------------
# ResidualConnection
# ---------------------

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
    
# ----------------------
# FeedForwardBlock
# ----------------------

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

# -------------------------------------------------
# ProjectionLayer(nn.Module) - to get probability distro at decoder end
# --------------------------------------------------    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection(x)
    

# ------------------------------
# EncoderBlock (single encoder layer)
# ------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_bock = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mark):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mark))
        x = self.residual_connections[0](x, self.feed_forward_bock)
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
    
# -------------------------
# DecoderBlock (single decoder layer)
# -------------------------
      
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, 
                 cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.sel_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: MultiHeadAttention(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: MultiHeadAttention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)

        return x
    
# ------------------------------
# Decoder (full decoder network)
# ------------------------------

class Decoder(nn.Module):
    def __init__(self, features: int, layer: ):
        pass