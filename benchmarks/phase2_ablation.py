"""
Phase 2: Wavelet Ablation Benchmark (Go/No-Go 2)
Tests whether the 2D Wavelet transformation provides any inductive bias or empirical advantage
compared to a flat sequence representation with an identical parameter count.

Ablations:
  1. Vocoder Invertibility: SpecWave (with 2D DWT/IDWT) vs Flat (without Wavelets)
  2. Conditional Reasoner (Prompt -> Next Block):
     - Model A: SpecWave 2D Spectral Reasoner (maps spectral subbands)
     - Model B: Flat Spatial Reasoner (maps flattened spatial embeddings)
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_wave.wavelet import haar_dwt_2d, haar_idwt_2d
from benchmarks.common import get_device, set_seed, load_wikitext2_tokens

try:
    import tiktoken
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    sys.exit(1)


# =====================================================================
# Dataset for Sequence Pairs (Prompt -> Continuation)
# =====================================================================

class TokenPairDataset(Dataset):
    """
    Extracts contiguous pairs of (prompt_block, target_block) of fixed sequence length.
    """
    def __init__(self, token_list, seq_len=64, max_pairs=None, stride=32):
        self.pairs = []
        block_len = seq_len * 2
        for i in range(0, len(token_list) - block_len + 1, stride):
            prompt = token_list[i : i + seq_len]
            target = token_list[i + seq_len : i + block_len]
            self.pairs.append((prompt, target))
            if max_pairs is not None and len(self.pairs) >= max_pairs:
                break
                
        self.prompts = torch.tensor([p for p, t in self.pairs], dtype=torch.long)
        self.targets = torch.tensor([t for p, t in self.pairs], dtype=torch.long)
        
    def __len__(self):
        return len(self.prompts)
        
    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]


# =====================================================================
# Model Architectures
# =====================================================================

class SpecWaveConditionalReasoner(nn.Module):
    """
    SpecWave Model: Reasons purely in 2D Wavelet frequency domain.
    Prompt Tokens -> Embed -> 2D DWT -> Spectral MLP Reasoner -> 2D IDWT -> Refiner -> LM Head
    """
    def __init__(self, vocab_size=50257, seq_len=64, d_model=128, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        half_seq = seq_len // 2
        half_dim = d_model // 2
        spectral_dim = 4 * half_seq * half_dim # equals seq_len * d_model
        
        # Spectral Resonant Reasoner
        self.reasoner = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spectral_dim)
        )
        
        # Residual Wavelet Refiner
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, prompt_tokens):
        B = prompt_tokens.shape[0]
        half_seq = self.seq_len // 2
        half_dim = self.d_model // 2
        sub_size = half_seq * half_dim
        
        # 1. Embed & 2D DWT
        emb = self.embeddings(prompt_tokens) # [B, N, D]
        ll, lh, hl, hh = haar_dwt_2d(emb)
        spec_in = torch.cat([ll.flatten(1), lh.flatten(1), hl.flatten(1), hh.flatten(1)], dim=-1)
        
        # 2. Resonant Spectral Reasoner
        spec_out = self.reasoner(spec_in)
        
        # 3. Reshape to 4 Wavelet subbands
        o_ll = spec_out[:, 0 * sub_size : 1 * sub_size].view(B, half_seq, half_dim)
        o_lh = spec_out[:, 1 * sub_size : 2 * sub_size].view(B, half_seq, half_dim)
        o_hl = spec_out[:, 2 * sub_size : 3 * sub_size].view(B, half_seq, half_dim)
        o_hh = spec_out[:, 3 * sub_size : 4 * sub_size].view(B, half_seq, half_dim)
        
        # 4. 2D IDWT Synthesis
        reconstructed = haar_idwt_2d(o_ll, o_lh, o_hl, o_hh)
        
        # 5. Residual Refinement
        x_trans = reconstructed.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 6. Logits
        logits = self.lm_head(refined)
        return logits, refined


class FlatConditionalReasoner(nn.Module):
    """
    Ablation Baseline: Flat Sequence Reasoner WITHOUT Wavelets.
    Uses exact same parameter count and layer shapes, but operates directly on flattened embeddings.
    Prompt Tokens -> Embed -> Spatial MLP Reasoner -> Refiner -> LM Head
    """
    def __init__(self, vocab_size=50257, seq_len=64, d_model=128, hidden_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        flat_dim = seq_len * d_model
        
        # Spatial Reasoner (identical parameter dimensions to SpecWave)
        self.reasoner = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim)
        )
        
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, prompt_tokens):
        B = prompt_tokens.shape[0]
        
        # 1. Embed & Flatten
        emb = self.embeddings(prompt_tokens) # [B, N, D]
        flat_in = emb.flatten(1)
        
        # 2. Flat Spatial Reasoner
        flat_out = self.reasoner(flat_in)
        reconstructed = flat_out.view(B, self.seq_len, self.d_model)
        
        # 3. Residual Refinement
        x_trans = reconstructed.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 4. Logits
        logits = self.lm_head(refined)
        return logits, refined


# =====================================================================
# Evaluation & Training Routine
# =====================================================================

def evaluate_reasoner(model, dataloader, device, max_eval_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct_tokens = 0
    total_exact_seq_match = 0
    total_seqs = 0
    
    with torch.no_grad():
        for i, (prompts, targets) in enumerate(dataloader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            prompts, targets = prompts.to(device), targets.to(device)
            logits, _ = model(prompts)
            
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1), reduction='sum')
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets)
            total_correct_tokens += correct.sum().item()
            total_tokens += targets.numel()
            
            seq_match = (correct.sum(dim=-1) == targets.shape[1])
            total_exact_seq_match += seq_match.sum().item()
            total_seqs += targets.shape[0]
            
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


def train_and_eval_model(model_cls, model_name, train_loader, test_loader, device, vocab_size, seq_len=64, d_model=128, epochs=3, lr=2e-3):
    set_seed(42)
    model = model_cls(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{model_name.upper()}] Initialized with {num_params:,} parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    print("-" * 90)
    print(f"{'Epoch':<6} | {'Step':<7} | {'Train Loss':<11} | {'Train PPL':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Tok Acc':<12}")
    print("-" * 90)
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            logits, _ = model(prompts)
            
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 100 == 0 or global_step == 1:
                val_metrics = evaluate_reasoner(model, test_loader, device, max_eval_batches=30)
                train_ppl = math.exp(min(loss.item(), 20.0))
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {loss.item():<11.4f} | {train_ppl:<10.2f} | "
                    f"{val_metrics['loss']:<10.4f} | {val_metrics['ppl']:<10.2f} | "
                    f"{val_metrics['token_acc']:>10.2f}%",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    final_val = evaluate_reasoner(model, test_loader, device, max_eval_batches=None)
    
    print(f"[{model_name.upper()} FINAL] Blind Val Loss: {final_val['loss']:.4f} | Val PPL: {final_val['ppl']:.2f} | Val Tok Acc: {final_val['token_acc']:.2f}% | Time: {elapsed:.2f}s", flush=True)
    return final_val


def run_phase2_ablation(max_train_pairs=6000, max_test_pairs=800, seq_len=64, d_model=128, epochs=3, batch_size=64):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 90)
    print("🔬 SPEC-WAVE PHASE 2: WAVELET ABLATION BENCHMARK (SPECWAVE vs FLAT)")
    print(f"Device: {device} | d_model: {d_model} | seq_len: {seq_len} | Epochs: {epochs}")
    print("=" * 90)
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    print("Loading WikiText-2 sequence pairs...", flush=True)
    train_tokens, test_tokens = load_wikitext2_tokens(tokenizer=None)
    
    train_ds = TokenPairDataset(train_tokens, seq_len=seq_len, max_pairs=max_train_pairs, stride=32)
    test_ds = TokenPairDataset(test_tokens, seq_len=seq_len, max_pairs=max_test_pairs, stride=64)
    
    print(f"Train Dataset: {len(train_ds):,} pairs ({len(train_ds) * seq_len * 2:,} tokens)", flush=True)
    print(f"Test Dataset:  {len(test_ds):,} pairs ({len(test_ds) * seq_len * 2:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # -------------------------------------------------------------
    # 1. Train & Eval Model A: SpecWave (2D Wavelets)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("🔷 EXPERIMENT A: SPECWAVE (2D Wavelet Subband Resonant Reasoner)")
    print("=" * 90)
    results_a = train_and_eval_model(
        SpecWaveConditionalReasoner, "SpecWave (2D DWT)",
        train_loader, test_loader, device, vocab_size,
        seq_len=seq_len, d_model=d_model, epochs=epochs
    )
    
    # -------------------------------------------------------------
    # 2. Train & Eval Model B: Flat Baseline (No Wavelets)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("🔶 EXPERIMENT B: FLAT BASELINE (Spatial Reasoner without Wavelets)")
    print("=" * 90)
    results_b = train_and_eval_model(
        FlatConditionalReasoner, "Flat Baseline (No Wavelets)",
        train_loader, test_loader, device, vocab_size,
        seq_len=seq_len, d_model=d_model, epochs=epochs
    )
    
    # -------------------------------------------------------------
    # Summary & Decision Gate 2
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("📊 PHASE 2 ABLATION COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Model':<30} | {'Blind Val Loss':<16} | {'Blind Val PPL':<16} | {'Val Token Acc':<16}")
    print("-" * 90)
    print(f"{'Model A (SpecWave 2D Wavelet)':<30} | {results_a['loss']:<16.4f} | {results_a['ppl']:<16.2f} | {results_a['token_acc']:>14.2f}%")
    print(f"{'Model B (Flat Baseline)':<30} | {results_b['loss']:<16.4f} | {results_b['ppl']:<16.2f} | {results_b['token_acc']:>14.2f}%")
    print("-" * 90)
    
    diff_loss = results_b['loss'] - results_a['loss']
    diff_ppl = results_b['ppl'] - results_a['ppl']
    
    print(f"PPL Difference (Flat - SpecWave): {diff_ppl:+.2f} (Loss Diff: {diff_loss:+.4f})")
    
    if results_a['ppl'] < results_b['ppl'] * 0.90:
        gate_status = "✅ PASS: Wavelets provide significant inductive advantage (>10% better PPL)."
    elif abs(results_a['ppl'] - results_b['ppl']) / max(results_a['ppl'], results_b['ppl']) < 0.05:
        gate_status = "⚠️ NEUTRAL (B ~ A): Wavelets are functionally equivalent to flat representations (orthogonal reparametrization)."
    else:
        gate_status = "❌ NO-GO: Flat representation performs comparably or better."
        
    print(f"\nGate 2 Decision: {gate_status}")
    print("=" * 90)
    
    return {"specwave": results_a, "flat": results_b}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Wavelet Ablation")
    parser.add_argument("--max_train_pairs", type=int, default=6000)
    parser.add_argument("--max_test_pairs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    args = parser.parse_args()
    
    run_phase2_ablation(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model
    )
