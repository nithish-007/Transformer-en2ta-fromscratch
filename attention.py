import torch 
from torch import nn
import math 

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
