"""
Loss Ablation Study across 3 Seeds (Audit Item R3a / A5)
Evaluates whether auxiliary MSE losses (Spectral Parseval MSE / Manifold Embedding MSE)
help or hurt Cross-Entropy likelihood optimization in Native SpecWave LM.

Configurations:
  1. CE_ONLY: Pure Cross-Entropy Loss on tokens.
  2. CE_PLUS_SPECTRAL: CE + Multi-Scale Parseval MSE (LL, LH, HL, HH subbands).
  3. FULL_HYBRID: CE + Multi-Scale Parseval MSE + Manifold Embedding MSE.

Runs each configuration across 3 random seeds (42, 123, 999) with identical data budget.
Reports Mean ± Standard Deviation for Val Loss, Val PPL, and Token Accuracy.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_wave.native_model import NativeSpecWaveLM
from spec_wave.wavelet import haar_dwt_2d
from benchmarks.common import get_device, set_seed
from examples.train_tinystories_streaming_specwave import TinyStoriesStreamingDataset
from examples.train_native_specwave_lm import evaluate_native


def train_single_run(
    config_name,
    seed,
    train_ds,
    test_ds,
    device,
    epochs=1,
    batch_size=32,
    d_model=384,
    num_layers=6,
    nhead=6,
    lr=6e-4
):
    set_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = NativeSpecWaveLM(
        vocab_size=50257,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=64,
        target_len=64
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.98))
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    t0 = time.time()
    model.train()
    
    for epoch in range(1, epochs + 1):
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            
            if config_name == "FULL_HYBRID":
                total_loss, ce_loss, logits = model(prompts, targets)
            else:
                # Forward pass
                logits, (t_ll, t_lh, t_hl, t_hh) = model(prompts)
                ce_loss = F.cross_entropy(logits.reshape(-1, 50257), targets.reshape(-1))
                
                if config_name == "CE_ONLY":
                    total_loss = ce_loss
                elif config_name == "CE_PLUS_SPECTRAL":
                    with torch.no_grad():
                        gt_target_emb = model.wte(targets)
                        gt_ll, gt_lh, gt_hl, gt_hh = haar_dwt_2d(gt_target_emb)
                        
                    loss_ll = F.mse_loss(t_ll, gt_ll)
                    loss_hf = F.mse_loss(t_lh, gt_lh) + F.mse_loss(t_hl, gt_hl) + F.mse_loss(t_hh, gt_hh)
                    spectral_loss = 4.0 * loss_ll + 1.0 * loss_hf
                    total_loss = ce_loss + 2.0 * spectral_loss
                    
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
    elapsed = time.time() - t0
    val_metrics = evaluate_native(model, test_loader, device, max_batches=None)
    val_metrics["time"] = elapsed
    return val_metrics


def run_ablation_suite(
    max_train_pairs=6000,
    max_test_pairs=500,
    seeds=(42, 123, 999),
    batch_size=32
):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 95)
    print("🔬 SPECWAVE LOSS ABLATION STUDY (3 SEEDS x 3 CONFIGURATIONS)")
    print(f"Device: {device} | Train Tokens / Seed: {max_train_pairs * 128:,} | Seeds: {seeds}")
    print("=" * 95)
    
    print("Streaming TinyStories dataset...", flush=True)
    train_ds = TinyStoriesStreamingDataset(split='train', max_pairs=max_train_pairs, seq_len=64)
    test_ds = TinyStoriesStreamingDataset(split='validation', max_pairs=max_test_pairs, seq_len=64)
    
    configs = ["CE_ONLY", "CE_PLUS_SPECTRAL", "FULL_HYBRID"]
    results = {cfg: {"loss": [], "ppl": [], "acc": [], "time": []} for cfg in configs}
    
    print("\nStarting systematic ablation sweeps...\n", flush=True)
    print(f"{'Configuration':<20} | {'Seed':<6} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Token Acc':<14} | {'Time (s)'}")
    print("-" * 95)
    
    for cfg in configs:
        for seed in seeds:
            metrics = train_single_run(
                config_name=cfg,
                seed=seed,
                train_ds=train_ds,
                test_ds=test_ds,
                device=device,
                batch_size=batch_size
            )
            
            results[cfg]["loss"].append(metrics["loss"])
            results[cfg]["ppl"].append(metrics["ppl"])
            results[cfg]["acc"].append(metrics["acc"])
            results[cfg]["time"].append(metrics["time"])
            
            print(
                f"{cfg:<20} | {seed:<6d} | {metrics['loss']:<10.4f} | {metrics['ppl']:<10.2f} | "
                f"{metrics['acc']:>12.2f}% | {metrics['time']:>7.1f}s",
                flush=True
            )
            
    # -------------------------------------------------------------
    # Statistical Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("📊 LOSS ABLATION FINAL STATISTICAL SUMMARY (MEAN ± STD)")
    print("=" * 95)
    print(f"{'Configuration':<22} | {'Val Loss (Mean ± Std)':<24} | {'Val PPL (Mean ± Std)':<22} | {'Val Token Acc (Mean ± Std)'}")
    print("-" * 95)
    
    for cfg in configs:
        l_mean, l_std = np.mean(results[cfg]["loss"]), np.std(results[cfg]["loss"])
        p_mean, p_std = np.mean(results[cfg]["ppl"]), np.std(results[cfg]["ppl"])
        a_mean, a_std = np.mean(results[cfg]["acc"]), np.std(results[cfg]["acc"])
        
        print(
            f"{cfg:<22} | {l_mean:6.4f} ± {l_std:6.4f}          | {p_mean:6.2f} ± {p_std:6.2f}        | "
            f"{a_mean:5.2f}% ± {a_std:4.2f}%"
        )
    print("=" * 95)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loss Ablation Study across Seeds")
    parser.add_argument("--max_train_pairs", type=int, default=6000)
    parser.add_argument("--max_test_pairs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    run_ablation_suite(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        seeds=(42, 123, 999),
        batch_size=args.batch_size
    )
