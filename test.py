import torch
import torch.nn as nn
from tqdm import tqdm
import json
import argparse
import nltk
import yaml

from model import build_transformer
from data_loader import create_dataloaders
from train import greedy_decode, beam_search_decode


def test_model(checkpoint_path, config_path="config.yaml", num_examples=5, 
               use_beam_search=True, beam_size=None):
    """
    Test the trained model on test set
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration YAML file
        num_examples: Number of examples to print
        use_beam_search: Whether to use beam search
        beam_size: Beam size for beam search (None = use config value)
    
    Returns:
        BLEU score
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    dataset_name = config['dataset_name']
    src_tokenizer_path = config['src_tokenizer_path']
    tgt_tokenizer_path = config['tgt_tokenizer_path']
    seq_len = config['seq_len']
    d_model = config['d_model']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_ff = config['d_ff']
    dropout = config['dropout']
    tokenizer_type = config['tokenizer_type']
    
    if beam_size is None:
        beam_size = config['beam_size']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    _, _, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(
        dataset_name=dataset_name,
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_path=tgt_tokenizer_path,
        seq_len=seq_len,
        batch_size=1,
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
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    if 'bleu_score' in checkpoint:
        print(f"Model's BLEU score: {checkpoint['bleu_score']:.4f}")
    
    # Test
    print("\n" + "="*80)
    print(f"TESTING ({'Beam Search' if use_beam_search else 'Greedy Decoding'})")
    print("="*80)
    
    source_texts = []
    expected = []
    predicted = []
    
    console_width = 80
    example_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            enc_input = batch["enc_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            
            # Generate prediction
            if use_beam_search:
                model_out = beam_search_decode(model, enc_input, src_mask, tgt_tokenizer, 
                                               seq_len, device, beam_size=beam_size)
            else:
                model_out = greedy_decode(model, enc_input, src_mask, tgt_tokenizer, 
                                         seq_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            pred_text = tgt_tokenizer.decode(model_out.cpu().tolist())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(pred_text)
            
            # Print examples
            if example_count < num_examples:
                print('-' * console_width)
                print(f"Source:     {source_text}")
                print(f"Target:     {target_text}")
                print(f"Prediction: {pred_text}")
                example_count += 1
    
    # Calculate BLEU score
    references = [[ref.split()] for ref in expected]
    candidates = [pred.split() for pred in predicted]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Total examples: {len(predicted)}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print("="*80)
    
    # Save results
    results = {
        'bleu_score': bleu_score,
        'method': 'beam_search' if use_beam_search else 'greedy',
        'beam_size': beam_size if use_beam_search else 1,
        'examples': []
    }
    
    for src, ref, pred in zip(source_texts, expected, predicted):
        results['examples'].append({
            'source': src,
            'reference': ref,
            'prediction': pred
        })
    
    results_path = 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return bleu_score


def interactive_translate(checkpoint_path, config_path="config.yaml", beam_size=None):
    """
    Interactive translation mode
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration YAML file
        beam_size: Beam size for beam search (None = use config value)
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    src_tokenizer_path = config['src_tokenizer_path']
    tgt_tokenizer_path = config['tgt_tokenizer_path']
    seq_len = config['seq_len']
    d_model = config['d_model']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    d_ff = config['d_ff']
    dropout = config['dropout']
    
    if beam_size is None:
        beam_size = config['beam_size']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizers
    from tokenizers import Tokenizer
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
    
    # Build model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "="*80)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*80)
    print("Enter English text to translate to Tamil")
    print("Type 'quit' or 'exit' to stop")
    print("="*80 + "\n")
    
    while True:
        text = input("\nEnglish: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        # Tokenize
        tokens = src_tokenizer.encode(text).ids
        sos_idx = src_tokenizer.token_to_id('[SOS]')
        eos_idx = src_tokenizer.token_to_id('[EOS]')
        pad_idx = src_tokenizer.token_to_id('[PAD]')
        
        # Create input
        num_padding = seq_len - len(tokens) - 2
        if num_padding < 0:
            print(f"Input too long! Maximum {seq_len-2} tokens allowed.")
            continue
        
        enc_input = torch.tensor(
            [[sos_idx] + tokens + [eos_idx] + [pad_idx] * num_padding],
            dtype=torch.long,
            device=device
        )
        
        # Create mask
        src_mask = (enc_input != pad_idx).unsqueeze(1).unsqueeze(2).int()
        
        # Translate
        with torch.no_grad():
            model_out = beam_search_decode(model, enc_input, src_mask, tgt_tokenizer, 
                                          seq_len, device, beam_size=beam_size)
        
        translation = tgt_tokenizer.decode(model_out.cpu().tolist())
        print(f"Tamil:   {translation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Translation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to print')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding instead of beam search')
    parser.add_argument('--beam-size', type=int, default=None,
                       help='Beam size for beam search (overrides config)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive translation mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_translate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            beam_size=args.beam_size
        )
    else:
        test_model(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_examples=args.num_examples,
            use_beam_search=not args.greedy,
            beam_size=args.beam_size
        )
