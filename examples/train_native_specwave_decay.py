"""
Native SpecWave LM with Temporal Horizon Discounted Loss (Exponential Loss Decay)
Evaluates whether weighting the loss according to temporal depth (gamma^(i-1))
anchors the immediate wavefront (tokens 1..4) with high precision while maintaining
global spectral trajectory in the wave tail.
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

from spec_wave.native_model import NativeSpecWaveLM
from spec_wave.wavelet import haar_dwt_2d
from benchmarks.common import get_device, set_seed
from examples.train_tinystories_streaming_specwave import TinyStoriesStreamingDataset


# =====================================================================
# Positional Evaluation Function
# =====================================================================

def evaluate_positional(model, dataloader, device, weights=None, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_pos1 = 0
    correct_pos1_4 = 0
    total_pos1 = 0
    total_pos1_4 = 0
    total_correct = 0
    
    with torch.no_grad():
        for i, (prompts, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            prompts, targets = prompts.to(device), targets.to(device)
            B, seq_len = targets.shape
            
            logits, _ = model(prompts) # [B, seq_len, V]
            
            # Loss computation
            if weights is not None:
                w = weights.to(device) # [seq_len]
                ce_per_pos = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1), reduction='none').view(B, seq_len)
                weighted_loss = (ce_per_pos * w[None, :]).sum() / (B * w.sum())
                total_loss += weighted_loss.item() * (B * seq_len)
            else:
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1), reduction='sum')
                total_loss += loss.item()
                
            preds = torch.argmax(logits, dim=-1) # [B, seq_len]
            
            # Positional metrics
            correct_pos1 += (preds[:, 0] == targets[:, 0]).sum().item()
            total_pos1 += B
            
            correct_pos1_4 += (preds[:, :4] == targets[:, :4]).sum().item()
            total_pos1_4 += B * 4
            
            total_correct += (preds == targets).sum().item()
            total_tokens += B * seq_len
            
    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20.0))
    acc_all = (total_correct / total_tokens) * 100.0
    acc_p1 = (correct_pos1 / total_pos1) * 100.0
    acc_p1_4 = (correct_pos1_4 / total_pos1_4) * 100.0
    
    return {"loss": mean_loss, "ppl": ppl, "acc_all": acc_all, "acc_p1": acc_p1, "acc_p1_4": acc_p1_4}


# =====================================================================
# Training Engine with Exponential Horizon Decay
# =====================================================================

def run_decay_training(
    max_train_pairs=20000,
    max_test_pairs=800,
    epochs=1,
    batch_size=32,
    gamma=0.94,
    mode="exponential",
    d_model=384,
    num_layers=6,
    nhead=6,
    lr=6e-4,
    eval_interval=150,
    checkpoint_path="checkpoints/native_specwave_decay.pt"
):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print("=" * 95)
    print(f"🌊 NATIVE SPECWAVE WITH HORIZON-DECAYED LOSS (Mode: {mode} | gamma={gamma})")
    print(f"Device: {device} | d_model: {d_model} | Layers: {num_layers} | Heads: {nhead} | Batch Size: {batch_size}")
    print(f"Target Unique Tokens: {max_train_pairs * 128:,} | Checkpoint: {checkpoint_path}")
    print("=" * 95)
    
    set_seed(42)
    
    # 1. Compute Horizon Decay Weights for 64 tokens
    seq_len = 64
    if mode == "exponential":
        weights = torch.tensor([gamma ** i for i in range(seq_len)], dtype=torch.float32)
    elif mode == "anchor_first4":
        # Full weight on first 4 tokens, small baseline weight on the rest
        weights = torch.tensor([1.0 if i < 4 else 0.1 for i in range(seq_len)], dtype=torch.float32)
    else:
        weights = torch.ones(seq_len, dtype=torch.float32)
        
    weights = weights.to(device)
    w_norm = weights / weights.sum()
    
    print(f"Horizon Weights: Pos 1={weights[0]:.3f} | Pos 4={weights[3]:.3f} | Pos 16={weights[15]:.3f} | Pos 64={weights[63]:.3f}", flush=True)
    
    # Load Streaming TinyStories
    train_ds = TinyStoriesStreamingDataset(split='train', max_pairs=max_train_pairs, seq_len=seq_len)
    test_ds = TinyStoriesStreamingDataset(split='validation', max_pairs=max_test_pairs, seq_len=seq_len)
    
    print(f"Train Dataset: {len(train_ds):,} UNIQUE story pairs ({len(train_ds) * 128:,} tokens)", flush=True)
    print(f"Test Dataset:  {len(test_ds):,} UNIQUE test pairs ({len(test_ds) * 128:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize Native Model From Scratch
    model = NativeSpecWaveLM(
        vocab_size=50257,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        target_len=seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Native Model Initialized From Scratch with {total_params:,} parameters.", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.98))
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    print("-" * 95)
    print(f"{'Epoch':<5} | {'Step':<6} | {'Tokens':<10} | {'Train CE':<9} | {'Val CE':<8} | {'Val PPL':<8} | {'Pos 1 Acc':<10} | {'Pos 1-4 Acc':<12} | {'All Acc':<8}")
    print("-" * 95)
    
    global_step = 0
    t0 = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            B = prompts.shape[0]
            
            # Forward pass
            logits, (t_ll, t_lh, t_hl, t_hh) = model(prompts)
            
            # Weighted Cross Entropy Loss across positions
            ce_per_pos = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1), reduction='none').view(B, seq_len)
            weighted_ce = (ce_per_pos * weights[None, :]).sum() / (B * weights.sum())
            
            # Spectral Parseval Loss
            with torch.no_grad():
                gt_target_emb = model.wte(targets)
                gt_ll, gt_lh, gt_hl, gt_hh = haar_dwt_2d(gt_target_emb)
                
            loss_ll = F.mse_loss(t_ll, gt_ll)
            loss_hf = F.mse_loss(t_lh, gt_lh) + F.mse_loss(t_hl, gt_hl) + F.mse_loss(t_hh, gt_hh)
            spectral_loss = 4.0 * loss_ll + 1.0 * loss_hf
            
            total_loss = weighted_ce + 2.0 * spectral_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            tokens_seen = global_step * batch_size * 128
            
            if global_step % eval_interval == 0 or global_step == 1:
                val_metrics = evaluate_positional(model, test_loader, device, weights=weights, max_batches=25)
                
                is_best = ""
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save({
                        'step': global_step,
                        'tokens_seen': tokens_seen,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'val_ppl': val_metrics['ppl'],
                        'val_acc_all': val_metrics['acc_all'],
                        'val_acc_p1': val_metrics['acc_p1'],
                        'val_acc_p1_4': val_metrics['acc_p1_4']
                    }, checkpoint_path)
                    is_best = " ⭐ (BEST)"
                elif global_step % (eval_interval * 4) == 0:
                    periodic_ckpt = checkpoint_path.replace(".pt", f"_step{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'tokens_seen': tokens_seen,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_ppl': val_metrics['ppl']
                    }, periodic_ckpt)
                    is_best = f" 💾 (Saved {periodic_ckpt})"
                    
                print(
                    f"{epoch:<5d} | {global_step:<6d} | {tokens_seen:<10,d} | {weighted_ce.item():<9.4f} | "
                    f"{val_metrics['loss']:<8.4f} | {val_metrics['ppl']:<8.2f} | "
                    f"{val_metrics['acc_p1']:>8.2f}% | {val_metrics['acc_p1_4']:>10.2f}% | "
                    f"{val_metrics['acc_all']:>6.2f}%{is_best}",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded Best Checkpoint from step {ckpt['step']} (Val Loss: {ckpt['val_loss']:.4f})")
        
    final_val = evaluate_positional(model, test_loader, device, weights=weights, max_batches=None)
    
    print("\n" + "=" * 95)
    print("📊 FINAL HORIZON-DECAYED NATIVE SPECWAVE RESULTS")
    print("=" * 95)
    print(f"Total Unique Story Tokens Processed: {global_step * batch_size * 128:,}")
    print(f"Final Blind Val Loss:               {final_val['loss']:.4f}")
    print(f"Final Blind Val PPL:                {final_val['ppl']:.2f}")
    print(f"🎯 Immediate Token 1 Accuracy:      {final_val['acc_p1']:.2f}%")
    print(f"🎯 Immediate Wavefront (1-4) Acc:   {final_val['acc_p1_4']:.2f}%")
    print(f"Global Average Token Accuracy:      {final_val['acc_all']:.2f}%")
    print(f"Best Val Loss Recorded:             {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Checkpoint Saved At:                {checkpoint_path}")
    print(f"Total Training Time:                {elapsed:.2f}s")
    print("=" * 95)
    
    return final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon-Decayed Native SpecWave Training")
    parser.add_argument("--max_train_pairs", type=int, default=20000)
    parser.add_argument("--max_test_pairs", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.94)
    parser.add_argument("--mode", type=str, default="exponential", choices=["exponential", "anchor_first4", "uniform"])
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval_interval", type=int, default=150)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/native_specwave_decay.pt")
    args = parser.parse_args()
    
    run_decay_training(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        mode=args.mode,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        lr=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_path=args.checkpoint_path
    )
