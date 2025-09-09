import torch
from torch import nn
import math

# ---------------------------------
# Input Embeddings
# ---------------------------------

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
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

# --------------------------------
# LayerNormalization
# --------------------------------

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1)

        # Normalize and apply learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
  
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

    
# -------------------------
# DecoderBlock (single decoder layer)
# -------------------------
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, 
                 cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x   
    
# ------------------------------
# Decoder (full decoder network)
# ------------------------------

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()  
        self.layers = layers
        self.norm = LayerNormalization(features) #features --> d_model

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

# -------------------------------
# Transformer 
# -------------------------------
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

# -------------------------------
# Build Transformer (assemble everything)
# -------------------------------

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                       d_model: int=512, N:int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) ->Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)

    # Create a encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model=d_model, h=h,
                                                          dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_block = EncoderBlock(features=d_model, self_attention_block=encoder_self_attention_block,
                                     feed_forward_block=feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)
    
    # Create a decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        decoder_block = DecoderBlock(features=d_model, self_attention_block= decoder_self_attention_block,
                                     cross_attention_block= decoder_cross_attention_block, feed_forward_block= feed_forward_block,
                                     dropout=dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(features=d_model, layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(features=d_model, layers=nn.ModuleList(decoder_blocks))

    # Create the projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos,
                              projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer

# # code to test the architecture
# if __name__ == "__main__":
#     model = build_transformer(1000, 1000, 50, 50)
#     print(model)