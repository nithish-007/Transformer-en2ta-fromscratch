import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def causal_mask(size):
    """
    Create a causal mask to prevent attention to future positions.
    Uses lower triangular matrix.
    
    Args:
        size (int): Sequence length
    
    Returns:
        torch.Tensor: Causal mask of shape (1, size, size)
    """
    mask = torch.tril(torch.ones((1, size, size)))
    return mask


class TranslationDataset(Dataset):
    """
    Dataset class for English-Tamil translation task
    """
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, seq_len):
        """
        Args:
            dataset: HuggingFace dataset of translation pairs
            src_tokenizer: Tokenizer for source language (English)
            tgt_tokenizer: Tokenizer for target language (Tamil)
            seq_len: Maximum sequence length
        """
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    
    def __len__(self):
        """
        Returns:
            int: Number of samples in dataset
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the sample

        Returns:
            dict: Dictionary with enc_input, dec_input, label, text, and masks
        """
        item = self.dataset[index]

        src_text = item["input"]      # English is the source (from 'input' column)
        tgt_text = item["output"]     # Tamil is the target (from 'output' column)

        # Tokenize: convert text to token IDs
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Calculate padding needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2    # Both [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1    # Only [SOS], as [EOS] will be part of label
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too small for {len(enc_input_tokens)} or {len(dec_input_tokens)} tokens.")
        
        # Encoder input: [SOS] + tokens + [EOS] + [PAD]...
        enc_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        
        # Decoder input: [SOS] + tokens + [PAD]...
        # Example: [SOS] I am doing great [PAD] [PAD]
        dec_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Label: tokens + [EOS] + [PAD]...
        # Example: I am doing great [EOS] [PAD] [PAD]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )        

        # Encoder mask: mask all padding tokens
        # Shape: (1, 1, seq_len)
        encoder_mask = (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        # Decoder mask: mask padding tokens AND future tokens (causal mask)
        # Shape: (1, seq_len, seq_len)
        decoder_mask = (dec_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0))

        return {
            "enc_input": enc_input,        # (seq_len,)
            "dec_input": dec_input,        # (seq_len,)
            "label": label,                # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_mask": encoder_mask,      # (1, 1, seq_len)
            "tgt_mask": decoder_mask       # (1, seq_len, seq_len)
        }


def get_all_sentences(dataset, lang):
    """
    Extracts all sentences from the dataset for the specified language

    Args:
        dataset: HuggingFace dataset
        lang: Language key ('input' for English or 'output' for Tamil)

    Returns:
        List[str]: List of sentences in the specified language
    """
    sentences = []
    max_len = 0
    
    for entry in dataset:
        sentence = entry[lang]
        
        # Remove punctuation for cleaner tokenization
        clean_sentence = "".join(ch for ch in sentence if ch.isalnum() or ch.isspace()) 
        sentences.append(clean_sentence)
        
        # Track max length
        words = clean_sentence.split()
        if len(words) > max_len:
            max_len = len(words)
    
    print(f"The maximum sequence length in {lang} is {max_len}")
    
    return sentences


def get_or_build_tokenizer(path, dataset, lang, tokenizer_type="BPE"):
    """
    Loads a tokenizer from disk if available, otherwise trains a new one

    Args:
        path: Path to load or save the tokenizer JSON file
        dataset: Dataset used to train tokenizer if needed
        lang: Language to tokenize ('english' or 'tamil')
        tokenizer_type: 'BPE' or 'word_level'

    Returns:
        Tokenizer: A trained or loaded tokenizer for the given language
    """
    path = Path(path)

    if path.exists():
        print(f"Loading existing tokenizer from {path}")
        tokenizer = Tokenizer.from_file(str(path))
        return tokenizer
    else:
        print(f"Building new {tokenizer_type} tokenizer for {lang}...")
        
        if tokenizer_type == "BPE":
            # Use BPE (Byte Pair Encoding) model
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                vocab_size=32000,
                min_frequency=2
            )

            tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
            tokenizer.save(str(path))
            print(f"Tokenizer saved to {path}")
            return tokenizer
        
        elif tokenizer_type == "word_level":
            # Use WordLevel tokenizer
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
                min_frequency=2
            )
            
            tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
            tokenizer.save(str(path))
            print(f"Tokenizer saved to {path}")
            return tokenizer
        
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def create_dataloaders(dataset_name, src_tokenizer_path, tgt_tokenizer_path, 
                       seq_len, batch_size, tokenizer_type="BPE"):
    """
    Creates training, validation, and test dataloaders

    Args:
        dataset_name: HuggingFace dataset name
        src_tokenizer_path: Path to source tokenizer file
        tgt_tokenizer_path: Path to target tokenizer file
        seq_len: Maximum sequence length
        batch_size: Batch size for training
        tokenizer_type: 'BPE' or 'word_level'

    Returns:
        tuple: (train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer)
    """
    
    # Load dataset from HuggingFace
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    # Use the train split
    full_dataset = dataset['train']
    
    # Build or load tokenizers
    src_tokenizer = get_or_build_tokenizer(src_tokenizer_path, full_dataset, "input", tokenizer_type)
    tgt_tokenizer = get_or_build_tokenizer(tgt_tokenizer_path, full_dataset, "output", tokenizer_type)
    
    print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")

    # Split the data: 90% train, 5% val, 5% test
    ds_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = ds_split["train"]
    
    temp_split = ds_split["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = temp_split["train"]
    test_dataset = temp_split["test"]

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create Dataset objects
    train_ds = TranslationDataset(train_dataset, src_tokenizer, tgt_tokenizer, seq_len)
    val_ds = TranslationDataset(val_dataset, src_tokenizer, tgt_tokenizer, seq_len)
    test_ds = TranslationDataset(test_dataset, src_tokenizer, tgt_tokenizer, seq_len)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, src_tok, tgt_tok = create_dataloaders(
        dataset_name="jarvisvasu/english-to-colloquial-tamil",
        src_tokenizer_path="tokenizer_en.json",
        tgt_tokenizer_path="tokenizer_ta.json",
        seq_len=150,
        batch_size=8,
        tokenizer_type="BPE"
    )
    
    print("\n" + "="*50)
    print("Testing dataloader...")
    batch = next(iter(train_loader))
    print(f"Encoder input shape: {batch['enc_input'].shape}")
    print(f"Decoder input shape: {batch['dec_input'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Source mask shape: {batch['src_mask'].shape}")
    print(f"Target mask shape: {batch['tgt_mask'].shape}")
    print(f"\nSample source text: {batch['src_text'][0]}")
    print(f"Sample target text: {batch['tgt_text'][0]}")
    print("="*50)
