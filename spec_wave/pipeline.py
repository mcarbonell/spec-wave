import torch
import torch.nn as nn
from .wavelet import haar_dwt_2d, haar_idwt_2d

class EndToEndSpectralPipeline(nn.Module):
    """
    End-to-End Spectral Pipeline: Wave-In ➔ Wave-Out.
    Ingests prompt tokens as 2D Wavelets, reasons in the frequency domain, and synthesizes target tokens in 1 step.
    """
    def __init__(self, vocab_size: int = 256, in_seq_len: int = 32, out_seq_len: int = 32, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        half_in_seq, half_in_dim = in_seq_len // 2, d_model // 2
        half_out_seq, half_out_dim = out_seq_len // 2, d_model // 2
        
        in_spectral_dim = 4 * half_in_seq * half_in_dim
        out_spectral_dim = 4 * half_out_seq * half_out_dim
        
        # Spectral Resonant Reasoner
        self.spectral_reasoner = nn.Sequential(
            nn.Linear(in_spectral_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, out_spectral_dim)
        )
        
        # Parallel De-quantizer Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. Encodes input tokens into 2D Wavelet Prompt Waveform.
        2. Reasons purely in the frequency domain.
        3. Inverts output thought wave into full token block in 1 step.
        """
        B = input_tokens.shape[0]
        half_in_seq, half_in_dim = self.in_seq_len // 2, self.d_model // 2
        half_out_seq, half_out_dim = self.out_seq_len // 2, self.d_model // 2
        
        # 1. WAVE-IN: Tokens -> Continuous Embeddings -> 2D DWT
        in_emb = self.embeddings(input_tokens)
        in_ll, in_lh, in_hl, in_hh = haar_dwt_2d(in_emb)
        in_spectral_vec = torch.cat([in_ll.flatten(1), in_lh.flatten(1), in_hl.flatten(1), in_hh.flatten(1)], dim=-1)
        
        # 2. PURE WAVE REASONING: Transform Input Wave -> Output Wave
        out_spectral_vec = self.spectral_reasoner(in_spectral_vec)
        
        # Reshape to 4 Output Wavelet Subbands
        subband_size = half_out_seq * half_out_dim
        out_ll = out_spectral_vec[:, 0 * subband_size : 1 * subband_size].view(B, half_out_seq, half_out_dim)
        out_lh = out_spectral_vec[:, 1 * subband_size : 2 * subband_size].view(B, half_out_seq, half_out_dim)
        out_hl = out_spectral_vec[:, 2 * subband_size : 3 * subband_size].view(B, half_out_seq, half_out_dim)
        out_hh = out_spectral_vec[:, 3 * subband_size : 4 * subband_size].view(B, half_out_seq, half_out_dim)
        
        # 3. WAVE-OUT: 2D IDWT Wavelet Inversion -> Token Logits (Parallel O(1))
        out_embeddings = haar_idwt_2d(out_ll, out_lh, out_hl, out_hh)
        logits = self.lm_head(out_embeddings)
        
        return logits, out_spectral_vec
