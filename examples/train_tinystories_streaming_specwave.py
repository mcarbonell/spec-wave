"""
Massive Streaming Non-Repeating Training on TinyStories for SpecWave GPT-2
Evaluates the Spectral Cross-Attention Transformer on a structured narrative domain
with continuous streaming of unique non-overlapping story pairs.
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
from benchmarks.common import get_device, set_seed
from examples.train_gpt2_spectral_transformer import DeepGPT2SpectralModel, evaluate

try:
    from transformers import GPT2LMHeadModel
    from datasets import load_dataset
    import tiktoken
except ImportError:
    print("Please install transformers, datasets, tiktoken: pip install transformers datasets tiktoken")
    sys.exit(1)


# =====================================================================
# Streaming TinyStories Dataset Builder
# =====================================================================

class TinyStoriesStreamingDataset(Dataset):
    """
    Streams TinyStories from HuggingFace, tokenizes with tiktoken,
    and packs into strictly non-overlapping, unique (prompt, target) pairs.
    """
    def __init__(self, split='train', max_pairs=10000, seq_len=64):
        self.pairs = []
        enc = tiktoken.get_encoding("gpt2")
        block_len = seq_len * 2
        
        print(f"Streaming and tokenizing {max_pairs:,} unique pairs from TinyStories ({split})...", flush=True)
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
        token_buffer = []
        story_count = 0
        
        for item in ds:
            text = item.get("text", "").strip()
            if len(text) < 30:
                continue
            tokens = enc.encode(text)
            token_buffer.extend(tokens)
            story_count += 1
            
            while len(token_buffer) >= block_len:
                prompt = token_buffer[:seq_len]
                target = token_buffer[seq_len:block_len]
                self.pairs.append((prompt, target))
                token_buffer = token_buffer[block_len:] # Advance by full block -> 100% non-overlapping
                
                if len(self.pairs) >= max_pairs:
                    break
            if len(self.pairs) >= max_pairs:
                break
                
        print(f"Extracted {len(self.pairs):,} unique non-overlapping story pairs from {story_count:,} stories.", flush=True)
        self.prompts = torch.tensor([p for p, t in self.pairs], dtype=torch.long)
        self.targets = torch.tensor([t for p, t in self.pairs], dtype=torch.long)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]


# =====================================================================
# Main Training Function
# =====================================================================

def run_tinystories_training(
    max_train_pairs=8000,
    max_test_pairs=600,
    epochs=1,
    batch_size=16,
    unfreeze_from=6,
    eval_interval=100
):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 95)
    print("📖 SPECWAVE MASSIVE STREAMING TRAINING ON TINYSTORIES")
    print(f"Device: {device} | Unfreeze from layer {unfreeze_from} | Batch Size: {batch_size}")
    print("=" * 95)
    
    set_seed(42)
    
    print("Loading pretrained GPT-2 (124M)...", flush=True)
    gpt2_full = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_full.eval()
    
    pre_trained_head_weights = gpt2_full.lm_head.weight.detach()
    gpt2_backbone = gpt2_full.transformer
    
    # Load Streaming TinyStories
    train_ds = TinyStoriesStreamingDataset(split='train', max_pairs=max_train_pairs, seq_len=64)
    test_ds = TinyStoriesStreamingDataset(split='validation', max_pairs=max_test_pairs, seq_len=64)
    
    print(f"Train Dataset: {len(train_ds):,} UNIQUE story pairs ({len(train_ds) * 128:,} tokens)", flush=True)
    print(f"Test Dataset:  {len(test_ds):,} UNIQUE test pairs ({len(test_ds) * 128:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Baseline Native GPT-2 on TinyStories
    print("\nMeasuring Baseline Native GPT-2 on TinyStories Blind Validation Split...", flush=True)
    baseline_losses = []
    with torch.no_grad():
        for prompts, targets in test_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            full_seq = torch.cat([prompts, targets], dim=1)
            out = gpt2_full(full_seq)
            target_logits = out.logits[:, 63:-1, :]
            loss = F.cross_entropy(target_logits.reshape(-1, 50257), targets.reshape(-1))
            baseline_losses.append(loss.item())
            if len(baseline_losses) >= 40:
                break
                
    native_gpt2_loss = sum(baseline_losses) / len(baseline_losses)
    native_gpt2_ppl = math.exp(native_gpt2_loss)
    print(f"⭐ Baseline Native GPT-2 124M TinyStories Val PPL: {native_gpt2_ppl:.2f} (Loss: {native_gpt2_loss:.4f})", flush=True)
    print("-" * 95)
    
    # Initialize Model
    model = DeepGPT2SpectralModel(
        gpt2_backbone, pre_trained_head_weights,
        unfreeze_layers_from=unfreeze_from, seq_len=64, d_model=768
    ).to(device)
    
    gpt2_trainable = [p for p in model.gpt2.parameters() if p.requires_grad]
    adapter_trainable = [p for n, p in model.named_parameters() if not n.startswith("gpt2.") and p.requires_grad]
    
    total_trainable = sum(p.numel() for p in gpt2_trainable) + sum(p.numel() for p in adapter_trainable)
    print(f"Trainable Parameters: {total_trainable:,} (GPT-2: {sum(p.numel() for p in gpt2_trainable):,} | Adapter: {sum(p.numel() for p in adapter_trainable):,})", flush=True)
    
    optimizer = torch.optim.AdamW([
        {'params': gpt2_trainable, 'lr': 4e-5, 'weight_decay': 1e-4},
        {'params': adapter_trainable, 'lr': 4e-4, 'weight_decay': 1e-4}
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
                val_metrics = evaluate(model, test_loader, device, max_batches=25)
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
    print("📊 FINAL TINYSTORIES STREAMING RESULTS")
    print("=" * 95)
    print(f"Total Unique Story Tokens Processed: {global_step * batch_size * 128:,}")
    print(f"Final Blind Val Loss:               {final_val['loss']:.4f}")
    print(f"Final Blind Val PPL:                {final_val['ppl']:.2f}")
    print(f"Final Blind Val Token Acc:          {final_val['acc']:.2f}%")
    print(f"Best Val Loss Recorded:             {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Native GPT-2 Target PPL:            {native_gpt2_ppl:.2f}")
    print(f"Total Training Time:                {elapsed:.2f}s")
    print("=" * 95)
    
    return final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming TinyStories SpecWave Training")
    parser.add_argument("--max_train_pairs", type=int, default=8000)
    parser.add_argument("--max_test_pairs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unfreeze_from", type=int, default=6)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()
    
    run_tinystories_training(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        unfreeze_from=args.unfreeze_from,
        eval_interval=args.eval_interval
    )
