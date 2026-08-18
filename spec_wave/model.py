import time
import torch
import torch.nn as nn
from .wavelet import haar_dwt_2d
from .vocoder import ParallelSpectralLanguageVocoder

class SpecWaveLanguageModel(nn.Module):
    """
    SpecWave Language Model:
    Generates entire response paragraphs as 2D Spectral Thought Waveforms in a single forward pass.
    """
    def __init__(self, vocab_size: int = 256, seq_len: int = 64, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        half_seq = seq_len // 2
        half_dim = d_model // 2
        
        # Latent Semantic Projector: maps latent state to 4 Wavelet Subbands
        self.latent_to_spectral = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 4 * half_seq * half_dim)
        )
        
        # Parallel Spectral Vocoder
        self.vocoder = ParallelSpectralLanguageVocoder(seq_len=seq_len, d_model=d_model, vocab_size=vocab_size)

    def extract_ground_truth_wavelets(self, target_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert target token sequence to ground-truth 2D Wavelet Subbands"""
        emb = self.token_embeddings(target_tokens)
        return haar_dwt_2d(emb)

    def single_shot_generate(self, thought_context: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Generate ALL N tokens in 1 single forward step (O(1)) and measure wall-clock latency.
        """
        B = thought_context.shape[0]
        half_seq = self.seq_len // 2
        half_dim = self.d_model // 2
        
        t0 = time.perf_counter()
        
        # 1. Project thought context into 4 spectral subbands (LL, LH, HL, HH)
        spectral_raw = self.latent_to_spectral(thought_context)
        spectral_4d = spectral_raw.view(B, 4, half_seq, half_dim)
        
        ll = spectral_4d[:, 0]
        lh = spectral_4d[:, 1]
        hl = spectral_4d[:, 2]
        hh = spectral_4d[:, 3]
        
        # 2. Parallel IDWT Vocoding: decode all N tokens simultaneously
        logits = self.vocoder(ll, lh, hl, hh)
        tokens = torch.argmax(logits, dim=-1)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return tokens, latency_ms
