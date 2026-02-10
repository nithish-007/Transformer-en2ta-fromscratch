---
title: Transformer-en2ta-fromscratch
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
---

# Transformer English to Tamil Translation - From Scratch

A complete PyTorch implementation of the Transformer architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper, applied to English-to-Tamil machine translation using real-world datasets.

## Overview

This project implements every component of the Transformer architecture without using `nn.Transformer`. All modules including multi-head attention, positional encoding, encoder-decoder blocks, and layer normalization are built from scratch to provide deep understanding of the architecture.

**Dataset:** jarvisvasu/english-to-colloquial-tamil from HuggingFace  
**Training:** 14,375 pairs | **Validation:** 799 pairs | **Test:** 799 pairs

---

## Features

- Pure PyTorch implementation without high-level abstractions
- Complete Transformer architecture with all components built from scratch
- BPE tokenization for both English and Tamil
- Training pipeline with TensorBoard logging
- Validation with BLEU score metrics
- Greedy decoding and beam search inference
- Early stopping mechanism
- Model checkpointing and best model tracking
- Configuration-based training via YAML files
- Interactive translation mode

---

## Architecture

**Model Configuration:**
- Encoder Layers: 6
- Decoder Layers: 6
- Attention Heads: 8
- Model Dimension: 512
- Feed-Forward Dimension: 2048
- Dropout: 0.1
- Sequence Length: 256
- Total Parameters: ~73M

**Components:**
- Sinusoidal Positional Encoding
- Multi-Head Scaled Dot-Product Attention
- Pre-Layer Normalization (Pre-LN) variant
- Residual Connections
- Feed-Forward Networks
- Causal Masking for decoder
- Xavier Uniform weight initialization

---

## Project Structure

```
pytorch-transformer-from-scratch/
│
├── src/                          # Source code modules
│   ├── model.py                  # Main Transformer class
│   ├── encoder.py                # Encoder and EncoderBlock
│   ├── decoder.py                # Decoder, DecoderBlock, ProjectionLayer
│   ├── utils.py                  # Attention, Embeddings, FFN, LayerNorm
│   └── data_loader.py            # Dataset, tokenizers, dataloaders
│
├── train.py                      # Training pipeline with validation
├── test.py                       # Testing and interactive translation
├── requirements.txt              # Python dependencies
│
├── checkpoints/                  # Model checkpoints (created during training)
├── logs/                         # TensorBoard logs (created during training)
│
├── config/                       # Training configuration file
│   └── config.yaml                   
│
├── token_files/                  # Tokenizer JSON files (auto-generated)
│   ├── tokenizer_en.json
│   └── tokenizer_ta.json
│
└── README.md                     # This file
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nithish-007/Transformers_from_scratch.git
cd pytorch-transformer-from-scratch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- torch >= 2.0.0
- datasets (HuggingFace)
- tokenizers
- nltk
- tensorboard
- tqdm
- pyyaml

---

## Quick Start

### Train the Model

```bash
python train.py
```

Training uses settings from `config.yaml` by default. To use a custom config:

```bash
python train.py --config custom_config.yaml
```

### Monitor Training

```bash
tensorboard --logdir=logs
```

### Test the Model

```bash
python test.py --checkpoint checkpoints/best_model.pt
```

### Interactive Translation

```bash
python test.py --checkpoint checkpoints/best_model.pt --interactive
```

### Testing with Greedy Decoding

```bash
python test.py --checkpoint checkpoints/best_model.pt --greedy --num-examples 10
```

### Testing with Custom Beam Size

```bash
python test.py --checkpoint checkpoints/best_model.pt --beam-size 10
```

---

## Configuration

Edit `config.yaml` to customize training parameters:

---

## Training Pipeline

The training pipeline includes:

1. **Data Loading:** Automatic tokenizer building and dataset splitting (90/5/5)
2. **Model Initialization:** Xavier uniform weight initialization
3. **Training Loop:** 
   - Cross-entropy loss with padding token masking
   - Adam optimizer with gradient clipping
   - TensorBoard logging every 10 steps
4. **Validation:** 
   - Runs after each epoch on subset of validation data
   - Both greedy and beam search decoding
   - BLEU score calculation
5. **Checkpointing:**
   - Saves checkpoint after every epoch
   - Tracks and saves best model based on beam search BLEU
6. **Early Stopping:**
   - Monitors validation BLEU score
   - Stops training if no improvement for N epochs

---

## Code Architecture

### Core Components

**1. model.py**
- `Transformer`: Main model class with encode/decode/project methods
- `build_transformer()`: Factory function to assemble all components

**2. encoder.py**
- `EncoderBlock`: Self-attention + FFN with residual connections
- `Encoder`: Stack of N encoder blocks with final layer normalization

**3. decoder.py**
- `DecoderBlock`: Self-attention + cross-attention + FFN
- `Decoder`: Stack of N decoder blocks
- `ProjectionLayer`: Maps decoder output to vocabulary logits

**4. utils.py**
- `MultiHeadAttention`: Scaled dot-product attention with multiple heads
- `EmbeddingLayer`: Token embeddings
- `SinusoidalPositionalEncoding`: Position embeddings
- `FeedForwardBlock`: Two-layer MLP with ReLU
- `LayerNormalization`: Layer normalization
- `ResidualConnection`: Residual wrapper with dropout

**5. data_loader.py**
- `TranslationDataset`: PyTorch dataset for translation pairs
- `causal_mask()`: Lower triangular mask for decoder
- `get_or_build_tokenizer()`: BPE tokenizer training/loading
- `create_dataloaders()`: Complete data pipeline

---

## Results

Training results will vary based on dataset size and hyperparameters. Monitor TensorBoard for:
- Training loss curves
- Validation BLEU scores (greedy and beam search)
- Learning progress over epochs

---

## References

- Vaswani et al., ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- Harvard NLP [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- PyTorch Documentation

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Acknowledgments

- Dataset: jarvisvasu/english-to-colloquial-tamil from HuggingFace
- Inspired by the original Transformer paper and various open-source implementations
- Built with PyTorch, HuggingFace datasets, and tokenizers libraries
 