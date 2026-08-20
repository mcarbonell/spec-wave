"""
Deep Spectral Transformer Adapter for GPT-2 with Multi-Scale Wavelet Guidance
Key Improvements:
1. Deep Backbone Co-adaptation: Unfreezes GPT-2 layers 6 to 11 (+ LayerNorm).
2. Spectral Transformer Reasoner: Replaces flat MLP with a 4-layer Bidirectional Cross-Attention Transformer.
3. Multi-Scale Parseval Guidance Loss: Ponderates LL subband (low-frequency semantics) heavily over high frequencies.
4. Differential Learning Rates: Lower LR for GPT-2 backbone (5e-5) and higher LR for spectral adapter (5e-4).
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
from benchmarks.common import get_device, set_seed, load_wikitext2_tokens
from benchmarks.phase2_ablation import TokenPairDataset

try:
    from transformers import GPT2Model, GPT2LMHeadModel
    import tiktoken
except ImportError:
    print("Please install transformers and tiktoken: pip install transformers tiktoken")
    sys.exit(1)


# =====================================================================
# 1. Spectral Transformer Cross-Attention Reasoner
# =====================================================================

class SpectralTransformerReasoner(nn.Module):
    """
    Bidirectional Transformer Reasoner that attends over prompt context
    and generates structured 2D Wavelet spectral patches.
    """
    def __init__(self, d_model=768, nhead=8, num_layers=4, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.half_seq = seq_len // 2 # 32 spectral positions
        
        # Learnable spectral position queries for the output wave
        self.query_pos_emb = nn.Parameter(torch.randn(1, self.half_seq, d_model) * 0.02)
        
        # Transformer Decoder Layer with Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Spectral Subband Heads: projects each half-seq position to (LL, LH, HL, HH)
        # Each position produces 4 subbands of dimension d_model / 2
        half_dim = d_model // 2
        self.to_ll = nn.Linear(d_model, half_dim)
        self.to_lh = nn.Linear(d_model, half_dim)
        self.to_hl = nn.Linear(d_model, half_dim)
        self.to_hh = nn.Linear(d_model, half_dim)

    def forward(self, prompt_hidden_states):
        """
        prompt_hidden_states: [B, seq_len, d_model] from GPT-2
        Returns: ll, lh, hl, hh of shape [B, half_seq, half_dim]
        """
        B = prompt_hidden_states.shape[0]
        queries = self.query_pos_emb.repeat(B, 1, 1) # [B, 32, d_model]
        
        # Cross-attend: Query output tokens attend to all prompt tokens
        spectral_hidden = self.transformer_decoder(tgt=queries, memory=prompt_hidden_states) # [B, 32, d_model]
        
        ll = self.to_ll(spectral_hidden)
        lh = self.to_lh(spectral_hidden)
        hl = self.to_hl(spectral_hidden)
        hh = self.to_hh(spectral_hidden)
        
        return ll, lh, hl, hh


# =====================================================================
# 2. Full GPT-2 Deep Spectral Architecture
# =====================================================================

class DeepGPT2SpectralModel(nn.Module):
    def __init__(self, gpt2_backbone, pre_trained_head_weight, unfreeze_layers_from=6, seq_len=64, d_model=768):
        super().__init__()
        self.gpt2 = gpt2_backbone
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. Freeze early layers, UNFREEZE deep layers (e.g. layers 6-11)
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        if hasattr(self.gpt2, 'h'):
            num_blocks = len(self.gpt2.h)
            for i in range(unfreeze_layers_from, num_blocks):
                for param in self.gpt2.h[i].parameters():
                    param.requires_grad = True
        if hasattr(self.gpt2, 'ln_f'):
            for param in self.gpt2.ln_f.parameters():
                param.requires_grad = True
                
        # 2. Spectral Cross-Attention Transformer Reasoner
        self.spectral_reasoner = SpectralTransformerReasoner(d_model=d_model, nhead=12, num_layers=4, seq_len=seq_len)
        
        # 3. Residual 1D Manifold Refiner
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        # 4. Weight-Tied Pre-Trained LM Head
        vocab_size = pre_trained_head_weight.shape[0]
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight.data.copy_(pre_trained_head_weight.data)
        self.lm_head.weight.requires_grad = True

    def forward(self, prompt_ids):
        # 1. GPT-2 Context Forward
        gpt_out = self.gpt2(input_ids=prompt_ids)
        prompt_hidden = gpt_out.last_hidden_state # [B, 64, 768]
        
        # 2. Spectral Transformer Cross-Attention -> 4 Subbands
        o_ll, o_lh, o_hl, o_hh = self.spectral_reasoner(prompt_hidden)
        
        # 3. Exact 2D IDWT Wavelet Inversion -> Continuous Manifold
        reconstructed_emb = haar_idwt_2d(o_ll, o_lh, o_hl, o_hh) # [B, 64, 768]
        
        # 4. Residual Syntactic Refiner
        x_trans = reconstructed_emb.transpose(1, 2)
        refined_trans = self.refiner[2](self.refiner[1](self.refiner[0](x_trans))) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 5. Parallel De-quantization Head
        logits = self.lm_head(refined)
        
        return logits, (o_ll, o_lh, o_hl, o_hh), refined


# =====================================================================
# 3. Evaluation Engine
# =====================================================================

def evaluate(model, dataloader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for i, (prompts, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            prompts, targets = prompts.to(device), targets.to(device)
            logits, _, _ = model(prompts)
            
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1), reduction='sum')
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_tokens += targets.numel()
            
    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20.0))
    acc = (total_correct / total_tokens) * 100.0
    return {"loss": mean_loss, "ppl": ppl, "acc": acc}


# =====================================================================
# 4. Training Engine
# =====================================================================

def run_experiment(max_train_pairs=3000, max_test_pairs=300, epochs=3, batch_size=16, unfreeze_from=6):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 95)
    print("🚀 EXPERIMENT: DEEP GPT-2 (LAYERS 6-11) + SPECTRAL TRANSFORMER REASONER")
    print(f"Device: {device} | Unfreezing GPT-2 from layer {unfreeze_from} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 95)
    
    set_seed(42)
    
    print("Loading pretrained GPT-2 (124M)...", flush=True)
    gpt2_full = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_full.eval()
    
    pre_trained_head_weights = gpt2_full.lm_head.weight.detach()
    gpt2_backbone = gpt2_full.transformer
    
    print("Loading WikiText-2 sequence pairs...", flush=True)
    train_tokens, test_tokens = load_wikitext2_tokens(tokenizer=None)
    
    train_ds = TokenPairDataset(train_tokens, seq_len=64, max_pairs=max_train_pairs, stride=32)
    test_ds = TokenPairDataset(test_tokens, seq_len=64, max_pairs=max_test_pairs, stride=64)
    
    print(f"Train Dataset: {len(train_ds):,} pairs ({len(train_ds) * 128:,} tokens)", flush=True)
    print(f"Test Dataset:  {len(test_ds):,} pairs ({len(test_ds) * 128:,} tokens)", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # -------------------------------------------------------------
    # Baseline Native GPT-2 PPL on Test Split
    # -------------------------------------------------------------
    print("\nMeasuring Baseline Native GPT-2 Perplexity on Test Split...", flush=True)
    baseline_losses = []
    with torch.no_grad():
        for prompts, targets in test_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            full_seq = torch.cat([prompts, targets], dim=1)
            out = gpt2_full(full_seq)
            target_logits = out.logits[:, 63:-1, :]
            loss = F.cross_entropy(target_logits.reshape(-1, 50257), targets.reshape(-1))
            baseline_losses.append(loss.item())
            if len(baseline_losses) >= 30:
                break
                
    native_gpt2_loss = sum(baseline_losses) / len(baseline_losses)
    native_gpt2_ppl = math.exp(native_gpt2_loss)
    print(f"⭐ Baseline Native GPT-2 Causal Test PPL: {native_gpt2_ppl:.2f} (Loss: {native_gpt2_loss:.4f})", flush=True)
    print("-" * 95)
    
    # -------------------------------------------------------------
    # Initialize Deep Model
    # -------------------------------------------------------------
    model = DeepGPT2SpectralModel(
        gpt2_backbone, pre_trained_head_weights,
        unfreeze_layers_from=unfreeze_from, seq_len=64, d_model=768
    ).to(device)
    
    # Differential Parameters
    gpt2_trainable = [p for p in model.gpt2.parameters() if p.requires_grad]
    adapter_trainable = [p for n, p in model.named_parameters() if not n.startswith("gpt2.") and p.requires_grad]
    
    total_trainable = sum(p.numel() for p in gpt2_trainable) + sum(p.numel() for p in adapter_trainable)
    print(f"Trainable Parameters: {total_trainable:,} (GPT-2: {sum(p.numel() for p in gpt2_trainable):,} | Adapter: {sum(p.numel() for p in adapter_trainable):,})", flush=True)
    
    optimizer = torch.optim.AdamW([
        {'params': gpt2_trainable, 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': adapter_trainable, 'lr': 5e-4, 'weight_decay': 1e-4}
    ])
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Step':<7} | {'Train Loss':<11} | {'Train PPL':<10} | {'Val Loss':<10} | {'Val PPL':<10} | {'Val Tok Acc':<12}")
    print("-" * 95)
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            
            # Ground truth target embeddings & wavelets
            with torch.no_grad():
                target_embs = gpt2_full.transformer.wte(targets)
                gt_ll, gt_lh, gt_hl, gt_hh = haar_dwt_2d(target_embs)
                
            logits, (p_ll, p_lh, p_hl, p_hh), refined_embs = model(prompts)
            
            # 1. Cross-Entropy Loss
            ce_loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
            
            # 2. Multi-Scale Parseval Wavelet Loss (Heavy weight on LL semantic basin)
            loss_ll = F.mse_loss(p_ll, gt_ll)
            loss_hf = F.mse_loss(p_lh, gt_lh) + F.mse_loss(p_hl, gt_hl) + F.mse_loss(p_hh, gt_hh)
            spectral_loss = 4.0 * loss_ll + 1.0 * loss_hf
            
            # 3. Manifold Embedding MSE
            manifold_loss = F.mse_loss(refined_embs, target_embs)
            
            total_loss = ce_loss + 2.0 * spectral_loss + 2.0 * manifold_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 50 == 0 or global_step == 1:
                val_metrics = evaluate(model, test_loader, device, max_batches=20)
                train_ppl = math.exp(min(ce_loss.item(), 20.0))
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {ce_loss.item():<11.4f} | {train_ppl:<10.2f} | "
                    f"{val_metrics['loss']:<10.4f} | {val_metrics['ppl']:<10.2f} | "
                    f"{val_metrics['acc']:>10.2f}%",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    final_val = evaluate(model, test_loader, device, max_batches=None)
    
    print("\n" + "=" * 95)
    print("📊 FINAL DEEP SPECTRAL TRANSFORMER RESULTS")
    print("=" * 95)
    print(f"Blind Test Loss:         {final_val['loss']:.4f}")
    print(f"Blind Test PPL:          {final_val['ppl']:.2f}")
    print(f"Blind Test Token Acc:    {final_val['acc']:.2f}%")
    print(f"Native GPT-2 Target PPL: {native_gpt2_ppl:.2f}")
    print(f"Total Training Time:     {elapsed:.2f}s")
    print("=" * 95)
    
    return final_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep GPT-2 Spectral Transformer")
    parser.add_argument("--max_train_pairs", type=int, default=2000)
    parser.add_argument("--max_test_pairs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--unfreeze_from", type=int, default=6)
    args = parser.parse_args()
    
    run_experiment(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        unfreeze_from=args.unfreeze_from
    )
