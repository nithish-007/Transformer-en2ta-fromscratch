import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import nltk
import yaml

from model import build_transformer
from data_loader import create_dataloaders


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_bleu = 0
    
    def __call__(self, bleu_score):
        """
        Call this after each epoch with validation BLEU score
        Returns True if training should stop
        """
        score = bleu_score
        
        if self.best_score is None:
            self.best_score = score
            self.best_bleu = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_bleu = score
            self.counter = 0
        
        return self.early_stop


def greedy_decode(model, src, src_mask, tgt_tokenizer, max_len, device):
    """
    Greedy decoding: always select the token with highest probability
    
    Args:
        model: Transformer model
        src: Source sequence (1, seq_len)
        src_mask: Source mask
        tgt_tokenizer: Target tokenizer
        max_len: Maximum length to generate
        device: Device to run on
    
    Returns:
        Generated sequence
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Start with SOS token
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    
    for _ in range(max_len):
        # Create decoder mask
        decoder_mask = torch.tril(torch.ones((1, 1, decoder_input.size(1), decoder_input.size(1)), device=device)).int()
        
        # Decode
        decoder_output = model.decode(decoder_input, decoder_mask, encoder_output, src_mask)
        
        # Project to vocabulary
        projection_output = model.project(decoder_output)
        
        # Get next token (greedy selection)
        next_token = projection_output[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Stop if EOS is generated
        if next_token.item() == eos_idx:
            break
    
    return decoder_input.squeeze(0)


def beam_search_decode(model, src, src_mask, tgt_tokenizer, max_len, device, beam_size=5):
    """
    Beam search decoding: maintain top-k candidates
    
    Args:
        model: Transformer model
        src: Source sequence (1, seq_len)
        src_mask: Source mask
        tgt_tokenizer: Target tokenizer
        max_len: Maximum length to generate
        device: Device to run on
        beam_size: Number of beams
    
    Returns:
        Best generated sequence
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Initialize beams with SOS token
    sequences = [[sos_idx]]
    scores = [0.0]
    
    for _ in range(max_len):
        all_candidates = []
        
        for i, seq in enumerate(sequences):
            if seq[-1] == eos_idx:
                all_candidates.append((scores[i], seq))
                continue
            
            decoder_input = torch.tensor([seq], dtype=torch.long, device=device)
            decoder_mask = torch.tril(torch.ones((1, 1, decoder_input.size(1), decoder_input.size(1)), device=device)).int()
            
            decoder_output = model.decode(decoder_input, decoder_mask, encoder_output, src_mask)
            projection_output = model.project(decoder_output)
            
            log_probs = torch.log_softmax(projection_output[:, -1, :], dim=-1)
            top_log_probs, top_indices = log_probs.topk(beam_size)
            
            for j in range(beam_size):
                candidate_seq = seq + [top_indices[0, j].item()]
                candidate_score = scores[i] + top_log_probs[0, j].item()
                all_candidates.append((candidate_score, candidate_seq))
        
        # Select top beam_size candidates
        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        sequences = [seq for score, seq in ordered[:beam_size]]
        scores = [score for score, seq in ordered[:beam_size]]
        
        if all(seq[-1] == eos_idx for seq in sequences):
            break
    
    return torch.tensor(sequences[0], dtype=torch.long, device=device)


def run_validation(model, val_loader, tgt_tokenizer, device, num_examples=2, num_batches=10):
    """
    Run validation on a subset of validation data
    
    Args:
        model: Transformer model
        val_loader: Validation data loader
        tgt_tokenizer: Target tokenizer
        device: Device
        num_examples: Number of examples to print
        num_batches: Number of batches to process
    
    Returns:
        BLEU score
    """
    model.eval()
    
    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_beam = []
    
    console_width = 80
    example_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            
            enc_input = batch["enc_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            
            # Greedy decoding
            model_out_greedy = greedy_decode(model, enc_input, src_mask, tgt_tokenizer, 150, device)
            
            # Beam search decoding
            model_out_beam = beam_search_decode(model, enc_input, src_mask, tgt_tokenizer, 150, device, beam_size=3)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            
            model_out_greedy_text = tgt_tokenizer.decode(model_out_greedy.cpu().tolist())
            model_out_beam_text = tgt_tokenizer.decode(model_out_beam.cpu().tolist())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted_greedy.append(model_out_greedy_text)
            predicted_beam.append(model_out_beam_text)
            
            # Print examples
            if example_count < num_examples:
                print('-' * console_width)
                print(f"Source: {source_text}")
                print(f"Target: {target_text}")
                print(f"Greedy: {model_out_greedy_text}")
                print(f"Beam:   {model_out_beam_text}")
                example_count += 1
    
    # Calculate BLEU scores
    references = [[ref.split()] for ref in expected]
    
    candidates_greedy = [pred.split() for pred in predicted_greedy]
    bleu_greedy = nltk.translate.bleu_score.corpus_bleu(references, candidates_greedy)
    
    candidates_beam = [pred.split() for pred in predicted_beam]
    bleu_beam = nltk.translate.bleu_score.corpus_bleu(references, candidates_beam)
    
    print(f"\nBLEU Score (Greedy): {bleu_greedy:.4f}")
    print(f"BLEU Score (Beam):   {bleu_beam:.4f}")
    
    return bleu_greedy, bleu_beam


def train_model(config_path="config.yaml"):
    """
    Main training function - loads configuration from YAML file
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    print("Loading configuration from", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    dataset_name = config['dataset_name']
    src_tokenizer_path = config['src_tokenizer_path']
    tgt_tokenizer_path = config['tgt_tokenizer_path']
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    d_model = config['d_model']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_ff = config['d_ff']
    dropout = config['dropout']
    learning_rate = config['learning_rate']
    tokenizer_type = config['tokenizer_type']
    checkpoint_dir = config['checkpoint_dir']
    early_stopping_patience = config['early_stopping_patience']
    early_stopping_min_delta = config['early_stopping_min_delta']
    
    print("\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {n_layers} layers, {d_model} dim, {n_heads} heads")
    print(f"  Training: {num_epochs} epochs, batch size {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Tokenizer: {tokenizer_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(
        dataset_name=dataset_name,
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_path=tgt_tokenizer_path,
        seq_len=seq_len,
        batch_size=batch_size,
        tokenizer_type=tokenizer_type
    )
    
    # Build model
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    model = build_transformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
        N=n_layers,
        h=n_heads,
        dropout=dropout,
        d_ff=d_ff
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    pad_idx = src_tokenizer.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-9)
    
    # TensorBoard
    writer = SummaryWriter(log_dir="logs")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                   min_delta=early_stopping_min_delta)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    global_step = 0
    best_bleu = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in batch_iterator:
            enc_input = batch["enc_input"].to(device)
            dec_input = batch["dec_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            encoder_output = model.encode(enc_input, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, dec_input, tgt_mask)
            projection_output = model.project(decoder_output)
            
            # Calculate loss
            loss = loss_fn(
                projection_output.view(-1, tgt_tokenizer.get_vocab_size()), 
                labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
            
            global_step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Average Training Loss: {avg_loss:.4f}")
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Run validation
        print(f"\nValidation for epoch {epoch+1}:")
        bleu_greedy, bleu_beam = run_validation(model, val_loader, tgt_tokenizer, device, 
                                                num_examples=2, num_batches=10)
        
        writer.add_scalar('Val/BLEU_Greedy', bleu_greedy, epoch)
        writer.add_scalar('Val/BLEU_Beam', bleu_beam, epoch)
        
        # Save best model based on beam search BLEU
        if bleu_beam > best_bleu:
            best_bleu = bleu_beam
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bleu_score': best_bleu
            }, best_model_path)
            print(f"✓ New best model saved! BLEU: {best_bleu:.4f}")
        
        # Early stopping check
        if early_stopping(bleu_beam):
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"Best BLEU score: {early_stopping.best_bleu:.4f}")
            break
    
    writer.close()
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"Best BLEU Score: {best_bleu:.4f}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Translation Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Train using config file
    train_model(config_path=args.config)
