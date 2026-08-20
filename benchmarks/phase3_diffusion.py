"""
Phase 3: Conditional Generation via Iterative Spectral Diffusion (Diffusion-LM in Wavelet Domain)
Evaluates whether iterative denoising in the continuous 2D Wavelet space overcomes
the multimodal collapse of single-shot NAR generation.

Pipeline:
  1. Target Block -> Embeddings -> 2D DWT -> Continuous Ground Truth Spectrum z_0 [B, 4, N/2, D/2]
  2. Add Gaussian noise: z_t = sqrt(alpha_bar_t)*z_0 + sqrt(1 - alpha_bar_t)*eps
  3. Denoiser predicts noise: eps_theta(z_t, t, prompt_context)
  4. DDIM / DDPM Sampling (10-20 steps) to reconstruct z_0
  5. 2D IDWT Vocoder decodes z_0 -> Token Logits
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
    import tiktoken
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    sys.exit(1)


# =====================================================================
# Sinusoidal Timestep Embeddings
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# =====================================================================
# Spectral Denoising Network (Denoiser Backbone)
# =====================================================================

class SpectralWaveletDenoiser(nn.Module):
    """
    Predicts Gaussian noise in the 2D Wavelet spectral representation z_t,
    conditioned on prompt context and diffusion timestep t.
    """
    def __init__(self, d_model=128, seq_len=64, hidden_dim=384, time_dim=128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.half_seq = seq_len // 2
        self.half_dim = d_model // 2
        self.spec_dim = 4 * self.half_seq * self.half_dim # seq_len * d_model
        
        # Timestep Projector
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prompt Context Encoder (Residual Bi-directional 1D CNN / MLP)
        self.prompt_proj = nn.Sequential(
            nn.Linear(self.spec_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spectral Residual Blocks
        self.in_proj = nn.Linear(self.spec_dim, hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, self.spec_dim)

    def forward(self, z_t_flat, t, prompt_spec_flat):
        # 1. Embed Time and Prompt
        t_emb = self.time_mlp(t) # [B, hidden_dim]
        p_emb = self.prompt_proj(prompt_spec_flat) # [B, hidden_dim]
        
        # 2. Input Projection + Conditioning
        h = self.in_proj(z_t_flat) + t_emb + p_emb
        
        # 3. Residual Blocks
        h = self.norm1(h + self.block1(h))
        h = self.norm2(h + self.block2(h))
        
        # 4. Predict Noise
        noise_pred = self.out_proj(h)
        return noise_pred


# =====================================================================
# Full Spectral Wavelet Diffusion System
# =====================================================================

class SpectralWaveletDiffusionLM(nn.Module):
    """
    End-to-End Spectral Diffusion Language Model:
    Learns continuous diffusion over 2D Wavelet subbands and decodes via parallel IDWT Vocoder.
    """
    def __init__(self, vocab_size=50257, seq_len=64, d_model=128, num_timesteps=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        
        self.half_seq = seq_len // 2
        self.half_dim = d_model // 2
        self.sub_size = self.half_seq * self.half_dim
        
        # Shared Continuous Embeddings & Frozen Vocoder LM Head
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        # Parallel Wavelet Vocoder Refiner & Head
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Denoising Backbone
        self.denoiser = SpectralWaveletDenoiser(d_model=d_model, seq_len=seq_len, hidden_dim=384)
        
        # Cosine Beta Schedule for Diffusion
        betas = self.cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def extract_wavelet_spectrum(self, token_ids):
        emb = self.embeddings(token_ids) # [B, N, D]
        ll, lh, hl, hh = haar_dwt_2d(emb)
        spec = torch.cat([ll.flatten(1), lh.flatten(1), hl.flatten(1), hh.flatten(1)], dim=-1)
        return spec

    def decode_spectrum_to_logits(self, spec_flat):
        B = spec_flat.shape[0]
        o_ll = spec_flat[:, 0 * self.sub_size : 1 * self.sub_size].view(B, self.half_seq, self.half_dim)
        o_lh = spec_flat[:, 1 * self.sub_size : 2 * self.sub_size].view(B, self.half_seq, self.half_dim)
        o_hl = spec_flat[:, 2 * self.sub_size : 3 * self.sub_size].view(B, self.half_seq, self.half_dim)
        o_hh = spec_flat[:, 3 * self.sub_size : 4 * self.sub_size].view(B, self.half_seq, self.half_dim)
        
        reconstructed = haar_idwt_2d(o_ll, o_lh, o_hl, o_hh)
        x_trans = reconstructed.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        logits = self.lm_head(refined)
        return logits

    def forward(self, prompt_tokens, target_tokens):
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        
        # 1. Ground Truth Continuous Spectra
        prompt_spec = self.extract_wavelet_spectrum(prompt_tokens)
        target_spec = self.extract_wavelet_spectrum(target_tokens)
        
        # 2. Sample random diffusion timesteps t ~ Uniform(0, T-1)
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # 3. Add noise according to schedule
        noise = torch.randn_like(target_spec)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        z_t = sqrt_alpha * target_spec + sqrt_one_minus_alpha * noise
        
        # 4. Predict noise with Denoiser
        pred_noise = self.denoiser(z_t, t, prompt_spec)
        
        # 5. Diffusion Loss (MSE in spectral space)
        diff_loss = F.mse_loss(pred_noise, noise)
        
        # Aux reconstruction loss on clean target
        recon_logits = self.decode_spectrum_to_logits(target_spec)
        ce_loss = F.cross_entropy(recon_logits.view(-1, self.vocab_size), target_tokens.view(-1))
        
        total_loss = diff_loss + 0.1 * ce_loss
        return total_loss, diff_loss, ce_loss

    @torch.no_grad()
    def sample_ddim(self, prompt_tokens, steps=10, eta=0.0):
        """
        Fast DDIM sampling from random Gaussian noise in 10-20 steps.
        """
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        prompt_spec = self.extract_wavelet_spectrum(prompt_tokens)
        
        # Start from pure Gaussian noise in 2D Wavelet space
        z = torch.randn(B, self.seq_len * self.d_model, device=device)
        
        times = torch.linspace(self.num_timesteps - 1, 0, steps).long().to(device)
        
        for i in range(len(times)):
            t = times[i].repeat(B)
            pred_noise = self.denoiser(z, t, prompt_spec)
            
            alpha_bar = self.alphas_cumprod[t[0]]
            prev_t = times[i + 1] if i + 1 < len(times) else torch.tensor(-1, device=device)
            alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
            
            # Predict z_0
            pred_z0 = (z - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            
            # Direction pointing to z_t_prev
            dir_z = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            z = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_z
            
        # Decode synthesized spectrum z_0 through Parallel 2D IDWT Vocoder
        logits = self.decode_spectrum_to_logits(z)
        tokens = torch.argmax(logits, dim=-1)
        return tokens, logits


# =====================================================================
# Benchmark & Validation Routine
# =====================================================================

def evaluate_diffusion(model, dataloader, device, ddim_steps=10, max_eval_batches=20):
    model.eval()
    total_diff_loss = 0.0
    total_tokens = 0
    total_correct = 0
    batches = 0
    
    with torch.no_grad():
        for i, (prompts, targets) in enumerate(dataloader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            prompts, targets = prompts.to(device), targets.to(device)
            
            _, diff_loss, _ = model(prompts, targets)
            total_diff_loss += diff_loss.item()
            
            # DDIM Generation
            gen_tokens, gen_logits = model.sample_ddim(prompts, steps=ddim_steps)
            correct = (gen_tokens == targets).sum().item()
            total_correct += correct
            total_tokens += targets.numel()
            batches += 1
            
    mean_diff_loss = total_diff_loss / batches
    tok_acc = (total_correct / total_tokens) * 100.0
    return {"diff_loss": mean_diff_loss, "token_acc": tok_acc}


def run_phase3_diffusion(max_train_pairs=4000, max_test_pairs=400, epochs=3, batch_size=32, d_model=128, seq_len=64, num_timesteps=50, ddim_steps=10):
    device = get_device()
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(16, os.cpu_count() or 8))
        
    print("=" * 90)
    print("🔬 SPEC-WAVE PHASE 3: SPECTRAL WAVELET DIFFUSION-LM (ITERATIVE REFINEMENT)")
    print(f"Device: {device} | d_model: {d_model} | seq_len: {seq_len} | Timesteps: {num_timesteps} | DDIM Steps: {ddim_steps}")
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
    
    model = SpectralWaveletDiffusionLM(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, num_timesteps=num_timesteps).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Diffusion System Initialized with {num_params:,} parameters.", flush=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    print("-" * 90)
    print(f"{'Epoch':<6} | {'Step':<7} | {'Total Loss':<12} | {'Diff Loss':<12} | {'Val Diff Loss':<15} | {'Val Gen Acc (10 DDIM)'}")
    print("-" * 90)
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for prompts, targets in train_loader:
            prompts, targets = prompts.to(device), targets.to(device)
            
            total_loss, diff_loss, ce_loss = model(prompts, targets)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 50 == 0 or global_step == 1:
                val_eval = evaluate_diffusion(model, test_loader, device, ddim_steps=ddim_steps, max_eval_batches=10)
                print(
                    f"{epoch:<6d} | {global_step:<7d} | {total_loss.item():<12.4f} | {diff_loss.item():<12.4f} | "
                    f"{val_eval['diff_loss']:<15.4f} | {val_eval['token_acc']:>18.2f}%",
                    flush=True
                )
                model.train()
                
    elapsed = time.time() - t0
    final_eval = evaluate_diffusion(model, test_loader, device, ddim_steps=ddim_steps, max_eval_batches=None)
    
    print("\n" + "=" * 90)
    print("📊 PHASE 3 FINAL EVALUATION (FULL BLIND TEST SPLIT)")
    print("=" * 90)
    print(f"Final Diffusion Noise MSE: {final_eval['diff_loss']:.4f}")
    print(f"Final 10-Step DDIM Gen Acc: {final_eval['token_acc']:.2f}%")
    print(f"Training Time:             {elapsed:.2f}s")
    print("=" * 90)
    
    return final_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Spectral Wavelet Diffusion")
    parser.add_argument("--max_train_pairs", type=int, default=3000)
    parser.add_argument("--max_test_pairs", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--ddim_steps", type=int, default=10)
    args = parser.parse_args()
    
    run_phase3_diffusion(
        max_train_pairs=args.max_train_pairs,
        max_test_pairs=args.max_test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_timesteps=args.num_timesteps,
        ddim_steps=args.ddim_steps
    )
