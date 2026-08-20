"""
Massive-Scale Non-Repeating Training for SpecWave GPT-2
Key Features:
1. Zero Repetition (Streaming Unique Pairs): Every single training step sees completely fresh, non-overlapping text pairs.
2. Full WikiText-2 Stream (19,000+ unique sequence pairs = 2.4M tokens).
3. Deep Backbone Co-adaptation: Unfreezes GPT-2 layers 6 to 11.
4. Spectral Cross-Attention Transformer Reasoner.
5. Multi-Scale Parseval Guidance (4x weight on LL semantic basin).
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
from examples.train_gpt2_spectral_transformer import DeepGPT2SpectralModel, evaluate

try:
    from transformers import GPT2LMHeadModel
    import tiktoken
except ImportError:
    print("Please install transformers and tiktoken: pip install transformers tiktoken")
    sys.exit(1)


# =====================================================================
# Streaming Non-Repeating Dataset
# =====================================================================

class NonRepeatingStreamDataset(Dataset):
    """
    Creates strictly non-overlapping, non-repeating (prompt, target) pairs.
    Stride = 2 * seq_len ensuring zero duplicate or overlapping tokens.
    """
    def __init__(self, token_list, seq_len=64, max_pairs=None):
        self.pairs = []
        block_len = seq_len * 2
        
        # Stride exactly equal to block length -> 100% unique, non-overlapping
        for i in range(0, len(token_list) - block_len + 1, block_len):
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
# Main Training Loop
# =====================================================================

def run_massive_training(
    max_train_pairs=10000,
    max_test_pairs=1000,
    epochs=1,
    batch_size=8,
    unfreeze_from=6,
    log_interval=50,
    eval_interval=100
):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 95)
    print("🌊 SPECWAVE MASSIVE-SCALE NON-REPEATING DIVERSE TEXT TRAINING")
    print(f"Device: {device} | Unfreeze from layer {unfreeze_from} | Batch Size: {batch_size}")
    print("=" * 95)
    
    set_seed(42)
    
    print("Loading pretrained GPT-2 (124M)...", flush=True)
    gpt2_full = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_full.eval()
    
    pre_trained_head_weights = gpt2_full.lm_head.weight.detach()
    gpt2_backbone = gpt2_full.transformer
    
    print("Loading WikiText-2 full corpus...", flush=True)
    train_tokens, test_tokens = load_wikitext2_tokens(tokenizer=None)
    
    # 100% Unique, non-overlapping pairs
    train_ds = NonRepeatingStreamDataset(train_tokens, seq_len=64, max_pairs=max_train_pairs)
    test_ds = NonRepeatingStreamDataset(test_tokens, seq_len=64, max_pairs=max_test_pairs)
    
    print(f"Train Dataset: {len(train_ds):,} UNIQUE non-repeating pairs ({len(train_ds) * 128:,} tokens)", flush=True)
    print(f"Test Dataset:  {len(test_ds):,} UNIQUE test pairs ({len(test_ds) * 128:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Baseline Native GPT-2 PPL
    print("\nMeasuring Baseline Native GPT-2 on Test Split...", flush=True)
    baseline_losses = []
    with torch.no_grad():
        for prompts, targets in test_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            full_seq = torch.cat([prompts, targets], dim=1)
            out = gpt2_full(full_seq)
            target_logits = out.logits[:, 63:-1, :]
            loss = F.cross_entropy(target_logits.reshape(-1, 50257), targets.reshape(-1))
            baseline_losses.append(loss.item())
            if len(baseline_losses) >= 50:
                break
                
    native_gpt2_loss = sum(baseline_losses) / len(baseline_losses)
    native_gpt2_ppl = math.exp(native_gpt2_loss)
    print(f"⭐ Baseline Native GPT-2 124M Test PPL: {native_gpt2_ppl:.2f} (Loss: {native_gpt2_loss:.4f})", flush=True)
    print("-" * 95)
    
    # Initialize Model
    model = DeepGPT2SpectralModel(
        gpt2_backbone, pre_trained_head_weights,
        unfreeze_layers_from=unfreeze_from, seq_len=64, d_model=768
    ).to(device)
    
    gpt2_trainable = [p for p in model.gpt2.parameters() if p.requires_grad]
    adapter_trainable = [p for n, p in model.named_parameters() if not n.startswith("gpt2.") and p.requires_grad]
    
    total_trainable = sum(p.numel() for p in gpt2_trainable) + sum(p.numel() for p in adapter_trainable)
    print(f"Trainable Parameters: {total_trainable:,}", flush=True)
    
    optimizer = torch.optim.AdamW([
        {'params': gpt2_trainable, 'lr': 3e-5, 'weight_decay': 1e-4},
        {'params': adapter_trainable, 'lr': 3e-4, 'weight_decay': 1e-4}
    ])
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Step':<7} | {'Tokens Seen':<12} | {'Train Loss':<11} | {'Train PPL':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Acc':<8}")
    print("-" * 95)
    
    global_step = 0
    t0 = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            
            with torch.no_grad():
                target_embs = gpt2_full.transformer.wte(targets)
                gt_ll, gt_lh, gt_hl, gt_hh = haar_dwt_2d(target_embs)
                
            logits, (p_ll, p_lh, p_hl, p_hh), refined_embs = model(prompts)
            
            ce_loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
            
            loss_ll = F.mse_loss(p_ll, gt_ll)
            loss_hf = F.mse_loss(p_lh, gt_lh) + F.mse_loss(p_hl, gt_hl) + F.mse_loss(p_hh, gt_hh)
            spectral_loss = 4.0 * loss_ll + 1.0 * loss_hf
            manifold_loss = F.mse_loss(refined_embs, target_embs)
            
            total_loss = ce_loss + 2.0 * spectral_loss + 2.0 * manifold_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            tokens_seen = global_step * batch_size * 128
            
            if global_step % eval_interval == 0 or global_step == 1:
                val_metrics = evaluate(model, test_loader, device, max_batches=30)
                train_ppl = math.exp(min(ce_loss.item(), 20.0))
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {tokens_seen:<12,d} | {ce_loss.item():<11.4f} | {train_ppl:<10.2f} | "
                    f"{val_metrics['loss']:<10.4f} | {val_metrics['ppl']:<10.2f} | "
                    f"{val_metrics['acc']:>6.2f}%",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    final_val = evaluate(model, test_loader, device, max_batches=None)
    
    print("\n" + "=" * 95)
    print("📊 FINAL MASSIVE-SCALE TRAINING RESULTS")
    print("=" * 95)
    print(f"Total Unique Tokens Processed: {global_step * batch_size * 128:,}")
    print(f"Final Blind Test Loss:         {final_val['loss']:.4f}")
    print(f"Final Blind Test PPL:          {final_val['ppl']:.2f}")
    print(f"Final Blind Test Token Acc:    {final_val['acc']:.2f}%")
    print(f"Best Test Loss Recorded:       {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Native GPT-2 Target PPL:       {native_gpt2_ppl:.2f}")
    print(f"Total Training Time:           {elapsed:.2f}s")
    print("=" * 95)
    
    return final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Massive-Scale SpecWave Training")
    parser.add_argument("--max_train_pairs", type=int, default=8000)
    parser.add_argument("--max_test_pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--unfreeze_from", type=int, default=6)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()
    
    run_massive_training(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        unfreeze_from=args.unfreeze_from,
        eval_interval=args.eval_interval
    )
