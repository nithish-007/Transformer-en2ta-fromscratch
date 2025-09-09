import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score
import numpy as np
import yaml
import time
import os
from data_loader import get_dataloaders
from engine import build_transformer

class TransformerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.src_vocab, self.tgt_vocab = get_dataloaders(config)
        
        # Build model
        self.model = build_transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            src_seq_len=config['seq_len'],
            tgt_seq_len=config['seq_len'],
            d_model=config['d_model'],
            N=config['n_layers'],
            h=config['n_heads'],
            dropout=config['dropout'],
            d_ff=config['d_ff']
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = Adam(self.model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab['<PAD>'])  # Ignore padding index
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, 
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Create reverse vocabularies for decoding
        self.src_idx_to_word = {idx: word for word, idx in self.src_vocab.items()}
        self.tgt_idx_to_word = {idx: word for word, idx in self.tgt_vocab.items()}
        
    def create_mask(self, src, tgt):
        # Create source mask (padding mask)
        src_mask = (src != self.src_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)
        
        # Create target mask (padding mask + look-ahead mask)
        tgt_pad_mask = (tgt != self.tgt_vocab['<PAD>']).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return src_mask, tgt_mask
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            
            # Create masks
            src_mask, tgt_mask = self.create_mask(src, tgt)
            
            # Forward pass
            self.optimizer.zero_grad()
            encoder_output = self.model.encode(src, src_mask)
            decoder_output = self.model.decode(encoder_output, src_mask, tgt[:, :-1], tgt_mask[:, :, :-1, :-1])
            output = self.model.project(decoder_output)
            
            # Calculate loss (ignore the first token which is SOS)
            loss = self.criterion(output.contiguous().view(-1, output.size(-1)), 
                                 tgt[:, 1:].contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
                self.writer.add_scalar('Training/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)
                
                # Log learning rate
                self.writer.add_scalar('Training/Learning Rate', self.optimizer.param_groups[0]['lr'], 
                                      epoch * len(self.train_loader) + batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_candidates = []
        all_references = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                # Create masks
                src_mask, tgt_mask = self.create_mask(src, tgt)
                
                # Forward pass
                encoder_output = self.model.encode(src, src_mask)
                decoder_output = self.model.decode(encoder_output, src_mask, tgt[:, :-1], tgt_mask[:, :, :-1, :-1])
                output = self.model.project(decoder_output)
                
                # Calculate loss
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), 
                                     tgt[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(output, dim=-1)
                
                # Convert indices to words for BLEU calculation
                for i in range(src.size(0)):
                    # Convert target indices to words
                    tgt_words = []
                    for idx in tgt[i]:
                        if idx.item() == self.tgt_vocab['<EOS>']:
                            break
                        if idx.item() not in [self.tgt_vocab['<PAD>'], self.tgt_vocab['<SOS>']]:
                            tgt_words.append(self.tgt_idx_to_word[idx.item()])
                    
                    # Convert predicted indices to words
                    pred_words = []
                    for idx in predicted[i]:
                        if idx.item() == self.tgt_vocab['<EOS>']:
                            break
                        if idx.item() not in [self.tgt_vocab['<PAD>'], self.tgt_vocab['<SOS>']]:
                            pred_words.append(self.tgt_idx_to_word[idx.item()])
                    
                    # Add to lists for BLEU calculation
                    all_candidates.append(pred_words)
                    all_references.append([tgt_words])  # Note: BLEU expects list of references per candidate
                
                # Print some samples
                if batch_idx % 50 == 0 and batch_idx == 0:
                    src_sentence = ' '.join([self.src_idx_to_word[idx.item()] for idx in src[0] 
                                           if idx.item() not in [self.src_vocab['<PAD>'], self.src_vocab['<SOS>'], self.src_vocab['<EOS>']]])
                    tgt_sentence = ' '.join([self.tgt_idx_to_word[idx.item()] for idx in tgt[0] 
                                           if idx.item() not in [self.tgt_vocab['<PAD>'], self.tgt_vocab['<SOS>'], self.tgt_vocab['<EOS>']]])
                    pred_sentence = ' '.join(pred_words)
                    
                    print(f"Source: {src_sentence}")
                    print(f"Target: {tgt_sentence}")
                    print(f"Predicted: {pred_sentence}")
                    print()
        
        # Calculate BLEU score
        bleu = bleu_score(all_candidates, all_references)
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/BLEU', bleu, epoch)
        
        # Update learning rate based on validation loss
        self.scheduler.step(avg_loss)
        
        print(f"Validation BLEU Score: {bleu:.4f}")
        
        return avg_loss, bleu
    
    def train(self):
        best_val_loss = float('inf')
        best_bleu = 0
        
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, bleu = self.validate(epoch)
            end_time = time.time()
            
            print(f'Epoch {epoch+1}/{self.config["num_epochs"]} | '
                  f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.4f} | '
                  f'Time: {end_time - start_time:.2f}s')
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'bleu_score': bleu,
                    'src_vocab': self.src_vocab,
                    'tgt_vocab': self.tgt_vocab,
                }, os.path.join(self.config['model_dir'], 'best_model_loss.pt'))
                print(f"Saved best model (loss) with validation loss: {val_loss:.4f}")
            
            # Also save best model based on BLEU score
            if bleu > best_bleu:
                best_bleu = bleu
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'bleu_score': bleu,
                    'src_vocab': self.src_vocab,
                    'tgt_vocab': self.tgt_vocab,
                }, os.path.join(self.config['model_dir'], 'best_model_bleu.pth'))
                print(f"Saved best model (BLEU) with BLEU score: {bleu:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'bleu_score': bleu,
                    'src_vocab': self.src_vocab,
                    'tgt_vocab': self.tgt_vocab,
                }, os.path.join(self.config['model_dir'], f'checkpoint_epoch_{epoch}.pth'))

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Train
    trainer = TransformerTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()