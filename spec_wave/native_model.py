"""
Native SpecWave Language Model (Trained Purely from Scratch)
An end-to-end spectral architecture designed from first principles:
  1. Native Continuous Spectral Embeddings (Co-designed for Wavelet Continuity).
  2. 2D Haar DWT Spectral Decomposition into Multi-Scale Subbands (LL, LH, HL, HH).
  3. Multiscale Wavelet-Transformer Reasoner with Cross-Attention.
  4. 2D Haar IDWT Vocoder + Residual Conv Refiner.
  5. Weight-Tied Output Head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spec_wave.wavelet import haar_dwt_2d, haar_idwt_2d


class NativeSpecWaveLM(nn.Module):
    """
    End-to-End Native SpecWave Language Model.
    Learns spectral language representations from scratch without pretrained weights.
    """
    def __init__(
        self,
        vocab_size=50257,
        d_model=384,
        nhead=6,
        num_layers=6,
        seq_len=64,
        target_len=64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.target_len = target_len
        
        self.prompt_half_seq = seq_len // 2
        self.target_half_seq = target_len // 2
        self.half_dim = d_model // 2
        
        # 1. Native Token & Positional Embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe_prompt = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.spectral_queries = nn.Parameter(torch.randn(1, self.target_half_seq, d_model) * 0.02)
        
        # 2. Prompt Spectral Projector
        self.prompt_spec_proj = nn.Linear(4 * self.half_dim, d_model)
        
        # 3. Multiscale Transformer Reasoner (Encoder-Decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)
        
        # 4. Wavelet Subband Emission Heads
        self.to_ll = nn.Linear(d_model, self.half_dim)
        self.to_lh = nn.Linear(d_model, self.half_dim)
        self.to_hl = nn.Linear(d_model, self.half_dim)
        self.to_hh = nn.Linear(d_model, self.half_dim)
        
        # 5. Native Residual 1D Refiner
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        # 6. Weight-Tied Output Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight # Native Weight Tying
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, prompt_tokens, target_tokens=None):
        B = prompt_tokens.shape[0]
        
        # 1. Native Embeddings + Positional
        prompt_emb = self.wte(prompt_tokens) + self.wpe_prompt # [B, 64, d_model]
        
        # 2. 2D Haar DWT on Prompt -> 4 Subbands
        p_ll, p_lh, p_hl, p_hh = haar_dwt_2d(prompt_emb) # each [B, 32, d_model/2]
        prompt_spec_cat = torch.cat([p_ll, p_lh, p_hl, p_hh], dim=-1) # [B, 32, 2*d_model]
        prompt_spec_tokens = self.prompt_spec_proj(prompt_spec_cat)   # [B, 32, d_model]
        
        # 3. Spectral Transformer Encoder on Prompt Waves
        memory = self.encoder(prompt_spec_tokens) # [B, 32, d_model]
        
        # 4. Spectral Transformer Decoder -> Target Waves
        queries = self.spectral_queries.repeat(B, 1, 1) # [B, 32, d_model]
        decoded_hidden = self.decoder(tgt=queries, memory=memory) # [B, 32, d_model]
        
        # 5. Emit 4 Subbands for Target
        t_ll = self.to_ll(decoded_hidden)
        t_lh = self.to_lh(decoded_hidden)
        t_hl = self.to_hl(decoded_hidden)
        t_hh = self.to_hh(decoded_hidden)
        
        # 6. Exact 2D Haar IDWT Reconstructs Target Embeddings in Parallel
        reconstructed_target_emb = haar_idwt_2d(t_ll, t_lh, t_hl, t_hh) # [B, 64, d_model]
        
        # 7. Residual 1D Refiner
        x_trans = reconstructed_target_emb.transpose(1, 2)
        refined_trans = self.refiner[2](self.refiner[1](self.refiner[0](x_trans))) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 8. Weight-Tied De-quantization
        logits = self.lm_head(refined) # [B, 64, vocab_size]
        
        if target_tokens is not None:
            # Training Losses
            ce_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_tokens.reshape(-1))
            
            # Ground truth spectral decomposition of target
            gt_target_emb = self.wte(target_tokens)
            gt_ll, gt_lh, gt_hl, gt_hh = haar_dwt_2d(gt_target_emb)
            
            # Multiscale Parseval Loss
            loss_ll = F.mse_loss(t_ll, gt_ll)
            loss_hf = F.mse_loss(t_lh, gt_lh) + F.mse_loss(t_hl, gt_hl) + F.mse_loss(t_hh, gt_hh)
            spectral_loss = 4.0 * loss_ll + 1.0 * loss_hf
            
            # Manifold Alignment Loss
            manifold_loss = F.mse_loss(refined, gt_target_emb)
            
            total_loss = ce_loss + 2.0 * spectral_loss + 2.0 * manifold_loss
            return total_loss, ce_loss, logits
            
        return logits, (t_ll, t_lh, t_hl, t_hh)
