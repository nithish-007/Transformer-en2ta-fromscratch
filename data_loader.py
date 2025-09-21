# ------------------
# dataset.py
# ------------------

from h11 import Data
from scipy import special
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np

class WordLevelTranslationDataset(Dataset):
    def __init__(self, dataset, src_vocab, tgt_vocab, seq_len=100):
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.seq_len = seq_len
        self.pad_token = '<PAD>'
        self.UNK_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS'

        # Create reverse vocabularies for decoding 
        self.src_idx_to_word = {idx: word for idx, word in self.src_vocab.items()}
        self.tgt_idx_to_word = {idx: word for idx, word in self.tgt_vocab.items()}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['en']
        tgt_text = item['ta']

        # Tokenize (word-level tokenization)
        src_tokens = src_text.lower().split()
        tgt_tokens = tgt_text.split() 

        # convert tokens to indices
        src_indices = [self.src_vocab.get(token, self.src_vocab["<UNK>"]) for token in src_tokens]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab["<UNK>"]) for token in tgt_tokens]

        # Add SOS and EOS tokens
        src_indices = [self.src_vocab["<SOS>"]] + src_indices[:self.seq_len-2] + [self.src_vocab["<EOS>"]]
        tgt_indices = [self.tgt_vocab["<SOS>"]] + tgt_indices[:self.seq_len-2] + [self.tgt_vocab["<EOS>"]]

        # PAD Sequence
        src_padding = [self.src_vocab[self.pad_token]] * (self.seq_len - len(src_indices))
        tgt_padding = [self.tgt_vocab[self.pad_token]] * (self.seq_len - len(tgt_indices))

        src_indices += src_padding
        tgt_indices += tgt_padding

        return {
            'src_ids': torch.tensor(src_indices, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_indices, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def build_vocabularies(dataset, min_freq=2):
    src_counter = Counter()
    tgt_counter = Counter()

    # Count word frequencies
    for item in dataset:
        src_text = item["en"]
        tgt_txt = item["ta"]

        # simple word-level tokenizer
        src_tokens = src_text.lower().split()
        tgt_tokens = tgt_txt.split()

        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)
    
    # Create vocabularies with special tokens
    special_tokens = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"]

    # Source vocabbulary (English)
    src_vocab = {}
    for word, count in src_counter.items():
        if count >= min_freq:
            src_vocab[word] = len(src_vocab)

    for token in special_tokens:
        src_vocab[token] = len(src_vocab)
    
    # target vocabulary (Tamil)
    tgt_vocab = {}
    for word, count in tgt_counter.items():
        if count >= min_freq:
            tgt_vocab[word] = len(tgt_vocab)

    for token in special_tokens:
        tgt_vocab[token] = len(tgt_vocab)
        
    return src_vocab, tgt_vocab   
     
def get_dataloaders(config):

    # load dataset from HuggingFace
    dataset = load_dataset(config["dataset_path"], split="train").train_test_split(test_size=0.2)

    # Build vocabularies from training data
    src_vocab, tgt_vocab = build_vocabularies(dataset["train"], min_freq = config["min_freq"])

    # Create datasets:
    train_dataset = WordLevelTranslationDataset(
        dataset["train"], src_vocab, tgt_vocab, config["seq_len"]
    )

    val_dataset = WordLevelTranslationDataset(
        dataset["test"], src_vocab, tgt_vocab, config["seq_len"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
