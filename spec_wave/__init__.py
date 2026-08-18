"""
SpecWave: Holistic Spectral Wave Language Synthesis & Parallel Vocoding Framework
"""

from .wavelet import haar_dwt_2d, haar_idwt_2d
from .vocoder import ParallelSpectralLanguageVocoder
from .model import SpecWaveLanguageModel
from .pipeline import EndToEndSpectralPipeline

__version__ = "0.1.0"
__all__ = [
    "haar_dwt_2d",
    "haar_idwt_2d",
    "ParallelSpectralLanguageVocoder",
    "SpecWaveLanguageModel",
    "EndToEndSpectralPipeline"
]
