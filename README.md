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

# 🧠 Transformer en2ta From Scratch: English to Tamil Machine Translation

This repository contains a complete **from-scratch implementation of the Transformer architecture** from the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), applied to a **real-world machine translation task**: English ➜ Tamil.

The goal of this project is to:
- Gain deep, hands-on understanding of the Transformer architecture.
- Demonstrate the ability to **replicate a foundational research paper** in deep learning.
- Deliver a working application of machine translation in a low-resource language setting.

---

## 📌 Features
- Pure PyTorch implementation (no `nn.Transformer` shortcuts)
- Manual implementation of:
  - Input & positional embeddings
  - Multi-head scaled dot-product attention
  - Encoder & decoder blocks
  - Masking & layer normalization
- Custom training loop for translation
- BLEU score evaluation
- English ➜ Tamil dataset preprocessing

---

## 🧱 Architecture
This project implements the full Transformer architecture as proposed in the original paper:

- 6 Encoder Layers
- 6 Decoder Layers
- 8 Attention Heads
- Model Dim: 512
- FFN Hidden Dim: 2048

---

## 📂 Folder Structure
```bash
.
├── data/                     # Raw and preprocessed data
├── models/                  # Model components (encoder, decoder, attention, etc.)
├── utils/                   # Tokenizers, BLEU scoring, masking utils
├── train.py                 # Training loop
├── eval.py                  # Evaluation script
├── inference.py             # Run translation from terminal
├── requirements.txt         # Python dependencies
└── README.md                # Project overview
```

---

## 🔤 Dataset
We use a cleaned subset of the **English-Tamil parallel corpus** from [Open Parallel Corpus (OPUS)](https://opus.nlpl.eu/).

- Sentences are tokenized and preprocessed.
- Byte Pair Encoding (BPE) or SentencePiece tokenizer used.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nithish-007/Transformers_from_scratch.git
cd Transformers_from_scratch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess data
```bash
python utils/preprocess.py --src en --tgt ta
```

### 4. Train the model
```bash
python train.py --epochs 20 --batch_size 64 --lr 1e-4
```

### 5. Evaluate
```bash
python eval.py
```

### 6. Translate
```bash
python inference.py --sentence "How are you?"
# Output: "நீங்கள் எப்படி இருக்கிறீர்கள்?"
```

---

## 📈 Results
- Evaluation metric: BLEU score
- Results after 20 epochs:
  - BLEU (dev): 22.5
  - BLEU (test): 21.3

---

## 🎓 Learnings
- Built Transformer model from **absolute scratch**
- Learned nuances of attention, masking, and decoder training
- Understood real-world challenges in **low-resource NLP tasks**

---

## 📚 References
- Vaswani et al., ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- Harvard NLP Annotated Transformer
- OpenNMT, Fairseq, and PyTorch source code

---

## 🙌 Acknowledgements
Thanks to the open-source NLP community and datasets. Special credit to the [OPUS corpus](https://opus.nlpl.eu/) for providing valuable multilingual data.

---

<!-- ## 📬 Contact
**Author:** Nithish Kumar  
**Mail:** itsmedecoder07@gmail.com -->

---

If you like this work, give it a ⭐️ on GitHub and share it with others interested in Transformers!

---

> 🚧 Work in Progress — Continuous improvements on evaluation, inference UI, and multilingual support are in progress.
