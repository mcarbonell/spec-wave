"""
Phase 1: Invertibility at Scale (Go/No-Go 1)
Evaluates whether the 2D Haar Wavelet Vocoder can reconstruct unseen text blocks from real WikiText-2.

Pipeline:
  Input Tokens [B, 64] -> Embeddings [B, 64, D] -> 2D DWT -> 2D IDWT -> Conv Refiner -> LM Head -> Logits [B, 64, V]

Compares:
  1. SpecWave Wavelet Vocoder (with 2D Haar DWT/IDWT)
  2. Flat Identity Vocoder (No Wavelets: direct Embed -> Conv Refiner -> LM Head)
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_wave.wavelet import haar_dwt_2d, haar_idwt_2d
from benchmarks.common import get_device, set_seed, TokenBlockDataset, load_wikitext2_tokens

try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("Error: transformers is required. pip install transformers")
    sys.exit(1)


# =====================================================================
# Models
# =====================================================================

class SpecWaveVocoderAutoencoder(nn.Module):
    """
    SpecWave 2D Wavelet Vocoder Autoencoder.
    Embeddings -> 2D Haar DWT (4 subbands) -> 2D Haar IDWT -> Conv1D Refiner -> LM Head
    """
    def __init__(self, vocab_size=50257, seq_len=64, d_model=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        # Spectral Refiner: 1D Convolutions over IDWT reconstructed manifold
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_tokens):
        # 1. Continuous Embedding: [B, seq_len, d_model]
        emb = self.embeddings(input_tokens)
        
        # 2. 2D Haar DWT Decomposition -> 4 Subbands
        ll, lh, hl, hh = haar_dwt_2d(emb)
        
        # 3. 2D Haar IDWT Synthesis -> Reconstructed embedding manifold
        reconstructed = haar_idwt_2d(ll, lh, hl, hh)
        
        # 4. Residual Syntactic Refiner
        x_trans = reconstructed.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 5. Parallel LM Head Logits
        logits = self.lm_head(refined)
        return logits, refined


class FlatVocoderAutoencoder(nn.Module):
    """
    Ablation Baseline: Flat Autoencoder WITHOUT Wavelet transformation.
    Embeddings -> Conv1D Refiner -> LM Head
    """
    def __init__(self, vocab_size=50257, seq_len=64, d_model=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_tokens):
        emb = self.embeddings(input_tokens)
        
        x_trans = emb.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        logits = self.lm_head(refined)
        return logits, refined


# =====================================================================
# Evaluation & Training Routine
# =====================================================================

def evaluate_model(model, dataloader, device, max_eval_batches=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_exact_seq_match = 0
    total_seqs = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            batch = batch.to(device)
            logits, _ = model(batch)
            
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), batch.view(-1), reduction='sum')
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == batch)
            total_correct_tokens += correct.sum().item()
            total_tokens += batch.numel()
            
            seq_match = (correct.sum(dim=-1) == batch.shape[1])
            total_exact_seq_match += seq_match.sum().item()
            total_seqs += batch.shape[0]
            
    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20.0))
    token_acc = (total_correct_tokens / total_tokens) * 100.0
    seq_acc = (total_exact_seq_match / total_seqs) * 100.0
    
    return {
        "loss": mean_loss,
        "ppl": ppl,
        "token_acc": token_acc,
        "seq_acc": seq_acc,
    }


def run_phase1_benchmark(
    max_train_blocks=50000,
    max_test_blocks=5000,
    seq_len=64,
    d_model=128,
    epochs=5,
    batch_size=32,
    lr=3e-3,
    stride=32,
    model_type="specwave"
):
    set_seed(42)
    device = get_device()
    print("=" * 90)
    print(f"🚀 SPEC-WAVE PHASE 1: VOCODER INVERTIBILITY AT SCALE ({model_type.upper()})")
    print(f"Device: {device} | d_model: {d_model} | seq_len: {seq_len} | Epochs: {epochs}")
    print("=" * 90)
    
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab # 50257
    
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
    
    print("Loading WikiText-2 real dataset...", flush=True)
    train_tokens, test_tokens = load_wikitext2_tokens(tokenizer=None)
    print(f"Loaded {len(train_tokens):,} train tokens and {len(test_tokens):,} test tokens.", flush=True)
    
    train_ds = TokenBlockDataset(train_tokens, seq_len=seq_len, max_blocks=max_train_blocks, stride=stride)
    test_ds = TokenBlockDataset(test_tokens, seq_len=seq_len, max_blocks=max_test_blocks, stride=seq_len)
    
    print(f"Train dataset: {len(train_ds):,} blocks ({len(train_ds) * seq_len:,} tokens)", flush=True)
    print(f"Test dataset:  {len(test_ds):,} blocks ({len(test_ds) * seq_len:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    if model_type == "specwave":
        model = SpecWaveVocoderAutoencoder(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model).to(device)
    else:
        model = FlatVocoderAutoencoder(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model).to(device)
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    print("-" * 90)
    print(f"{'Epoch':<6} | {'Step':<7} | {'Train Loss':<11} | {'Train PPL':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Tok Acc':<12} | {'Val Seq Match'}")
    print("-" * 90)
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 100 == 0 or global_step == 1:
                val_metrics = evaluate_model(model, test_loader, device, max_eval_batches=50)
                train_ppl = math.exp(min(loss.item(), 20.0))
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {loss.item():<11.4f} | {train_ppl:<10.2f} | "
                    f"{val_metrics['loss']:<10.4f} | {val_metrics['ppl']:<10.2f} | "
                    f"{val_metrics['token_acc']:>10.2f}% | {val_metrics['seq_acc']:>10.2f}%",
                    flush=True
                )
                model.train()
                
    total_time = time.time() - start_time
    
    # Final full evaluation on entire test split
    print("\n" + "=" * 90)
    print("📊 FINAL BLIND TEST EVALUATION (FULL TEST SPLIT)")
    print("=" * 90)
    final_test_metrics = evaluate_model(model, test_loader, device, max_eval_batches=None)
    
    print(f"Model Architecture:       {model_type.upper()}")
    print(f"Final Blind Test Loss:    {final_test_metrics['loss']:.4f}")
    print(f"Final Blind Test PPL:     {final_test_metrics['ppl']:.4f}")
    print(f"Final Token Accuracy:     {final_test_metrics['token_acc']:.2f}%")
    print(f"Final Exact Seq Match:    {final_test_metrics['seq_acc']:.2f}%")
    print(f"Total Training Time:      {total_time:.2f}s ({total_steps / total_time:.2f} steps/s)")
    
    decision = "✅ PASS / GO TO PHASE 2" if final_test_metrics['ppl'] <= 2.0 and final_test_metrics['token_acc'] >= 95.0 else "❌ NO-GO / FAILED GATE 1"
    print(f"\nDecision Gate (PPL <= 2.0 & Acc >= 95%): {decision}")
    print("=" * 90)
    
    return final_test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Vocoder Invertibility at Scale")
    parser.add_argument("--model_type", type=str, default="specwave", choices=["specwave", "flat"])
    parser.add_argument("--max_train_blocks", type=int, default=10000)
    parser.add_argument("--max_test_blocks", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    
    run_phase1_benchmark(
        max_train_blocks=args.max_train_blocks,
        max_test_blocks=args.max_test_blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        lr=args.lr,
        model_type=args.model_type
    )
