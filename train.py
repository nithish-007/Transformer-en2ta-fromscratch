import torch
from torch import nn
import math

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