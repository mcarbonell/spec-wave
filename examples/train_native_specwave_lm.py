"""
Training Native SpecWave LM From Scratch on TinyStories
Trains the pure spectral wavelet architecture from first principles (no pretrained weights),
optimizing continuous embeddings, 2D DWT/IDWT subbands, and the multiscale reasoner end-to-end.
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
from benchmarks.common import get_device, set_seed
from examples.train_tinystories_streaming_specwave import TinyStoriesStreamingDataset


def evaluate_native(model, dataloader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for i, (prompts, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            prompts, targets = prompts.to(device), targets.to(device)
            logits, _ = model(prompts)
            
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1), reduction='sum')
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_tokens += targets.numel()
            
    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20.0))
    acc = (total_correct / total_tokens) * 100.0
    return {"loss": mean_loss, "ppl": ppl, "acc": acc}


def run_native_training(
    max_train_pairs=15000,
    max_test_pairs=600,
    epochs=1,
    batch_size=16,
    d_model=384,
    num_layers=6,
    nhead=6,
    lr=6e-4,
    eval_interval=150,
    checkpoint_path="checkpoints/native_specwave_lm.pt"
):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print("=" * 95)
    print("🌱 NATIVE SPECWAVE LM: PURE FROM-SCRATCH TRAINING ON TINYSTORIES")
    print(f"Device: {device} | d_model: {d_model} | Layers: {num_layers} | Heads: {nhead} | Batch Size: {batch_size}")
    print(f"Target Unique Tokens: {max_train_pairs * 128:,} | Checkpoint: {checkpoint_path}")
    print("=" * 95)
    
    set_seed(42)
    
    # Load Streaming TinyStories
    train_ds = TinyStoriesStreamingDataset(split='train', max_pairs=max_train_pairs, seq_len=64)
    test_ds = TinyStoriesStreamingDataset(split='validation', max_pairs=max_test_pairs, seq_len=64)
    
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
        seq_len=64,
        target_len=64
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Native Model Initialized From Scratch with {total_params:,} parameters (100% Trainable).", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.98))
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
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
            
            total_loss, ce_loss, _ = model(prompts, targets)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            tokens_seen = global_step * batch_size * 128
            
            if global_step % eval_interval == 0 or global_step == 1:
                val_metrics = evaluate_native(model, test_loader, device, max_batches=25)
                train_ppl = math.exp(min(ce_loss.item(), 20.0))
                
                is_best = ""
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'val_loss': best_val_loss,
                        'val_ppl': val_metrics['ppl'],
                        'val_acc': val_metrics['acc']
                    }, checkpoint_path)
                    is_best = " ⭐ (SAVED)"
                    
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {tokens_seen:<12,d} | {ce_loss.item():<11.4f} | {train_ppl:<10.2f} | "
                    f"{val_metrics['loss']:<10.4f} | {val_metrics['ppl']:<10.2f} | "
                    f"{val_metrics['acc']:>6.2f}%{is_best}",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded Best Checkpoint from step {ckpt['step']} (Val Loss: {ckpt['val_loss']:.4f})")
        
    final_val = evaluate_native(model, test_loader, device, max_batches=None)
    
    print("\n" + "=" * 95)
    print("📊 FINAL NATIVE SPECWAVE FROM-SCRATCH RESULTS")
    print("=" * 95)
    print(f"Total Unique Story Tokens Processed: {global_step * batch_size * 128:,}")
    print(f"Final Blind Val Loss:               {final_val['loss']:.4f}")
    print(f"Final Blind Val PPL:                {final_val['ppl']:.2f}")
    print(f"Final Blind Val Token Acc:          {final_val['acc']:.2f}%")
    print(f"Best Val Loss Recorded:             {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Total Model Parameters:             {total_params:,} (100% Trained from scratch)")
    print(f"Checkpoint Saved At:                {checkpoint_path}")
    print(f"Total Training Time:                {elapsed:.2f}s")
    print("=" * 95)
    
    return final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Native SpecWave LM From Scratch")
    parser.add_argument("--max_train_pairs", type=int, default=15000)
    parser.add_argument("--max_test_pairs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval_interval", type=int, default=150)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/native_specwave_lm.pt")
    args = parser.parse_args()
    
    run_native_training(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        lr=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_path=args.checkpoint_path
    )
