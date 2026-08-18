import math
import torch

def haar_dwt_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D Discrete Haar Wavelet Transform on tensor of shape [B, H, W]
    Returns 4 subbands:
        - LL: Low-Low (Global semantic energy basin)
        - LH: Low-High (Horizontal syntactic transitions)
        - HL: High-Low (Vertical structural rhythm)
        - HH: High-High (Fine-grained lexical detail & noise)
    """
    row_low = (x[:, 0::2, :] + x[:, 1::2, :]) * (1.0 / math.sqrt(2.0))
    row_high = (x[:, 0::2, :] - x[:, 1::2, :]) * (1.0 / math.sqrt(2.0))
    
    ll = (row_low[:, :, 0::2] + row_low[:, :, 1::2]) * (1.0 / math.sqrt(2.0))
    lh = (row_low[:, :, 0::2] - row_low[:, :, 1::2]) * (1.0 / math.sqrt(2.0))
    hl = (row_high[:, :, 0::2] + row_high[:, :, 1::2]) * (1.0 / math.sqrt(2.0))
    hh = (row_high[:, :, 0::2] - row_high[:, :, 1::2]) * (1.0 / math.sqrt(2.0))
    
    return ll, lh, hl, hh


def haar_idwt_2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """
    Exact Lossless Inverse 2D Discrete Haar Wavelet Transform.
    Reconstructs continuous embedding tensor [B, H, W] from 4 subbands in a single parallel step.
    """
    B, H_half, W_half = ll.shape
    H = H_half * 2
    W = W_half * 2
    
    row_low = torch.zeros(B, H_half, W, device=ll.device, dtype=ll.dtype)
    row_low[:, :, 0::2] = (ll + lh) * (1.0 / math.sqrt(2.0))
    row_low[:, :, 1::2] = (ll - lh) * (1.0 / math.sqrt(2.0))
    
    row_high = torch.zeros(B, H_half, W, device=ll.device, dtype=ll.dtype)
    row_high[:, :, 0::2] = (hl + hh) * (1.0 / math.sqrt(2.0))
    row_high[:, :, 1::2] = (hl - hh) * (1.0 / math.sqrt(2.0))
    
    x = torch.zeros(B, H, W, device=ll.device, dtype=ll.dtype)
    x[:, 0::2, :] = (row_low + row_high) * (1.0 / math.sqrt(2.0))
    x[:, 1::2, :] = (row_low - row_high) * (1.0 / math.sqrt(2.0))
    
    return x
