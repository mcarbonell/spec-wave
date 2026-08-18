import torch
import torch.nn as nn
from .wavelet import haar_idwt_2d

class ParallelSpectralLanguageVocoder(nn.Module):
    """
    Parallel Spectral Language Vocoder:
    Synthesizes full blocks of token embeddings and projects to vocabulary logits in 1 single forward step (O(1)).
    """
    def __init__(self, seq_len: int = 64, d_model: int = 64, vocab_size: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Spectral Refiner: 1D Depthwise-Separable Convolutions over the IDWT reconstructed manifold
        self.refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model)
        )
        
        # Parallel De-quantizer Head: projects all N positions to token logits simultaneously
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        Input: 4 Wavelet Subbands of shape [B, seq_len/2, d_model/2]
        Output: Logits for all N tokens [B, seq_len, vocab_size] in a single GPU kernel.
        """
        # 1. Exact 2D IDWT Wavelet Inversion: [B, seq_len, d_model]
        reconstructed = haar_idwt_2d(ll, lh, hl, hh)
        
        # 2. Local syntactic manifold smoothing
        x_trans = reconstructed.transpose(1, 2)
        h = self.refiner[0](x_trans)
        h = self.refiner[1](h)
        refined_trans = self.refiner[2](h) + x_trans
        refined = self.refiner[3](refined_trans.transpose(1, 2))
        
        # 3. Parallel Token Logits: [B, seq_len, vocab_size]
        logits = self.lm_head(refined)
        return logits
