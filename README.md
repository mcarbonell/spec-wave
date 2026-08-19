# 🌊 SpecWave: Exploración de Síntesis de Lenguaje No-Autorregresiva vía Wavelets Espectrales 2D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**SpecWave** es un proyecto de investigación experimental que explora la generación de lenguaje no-autorregresiva (NAR): en lugar de generar tokens uno a uno ($O(N)$ pasos secuenciales), propone formular la generación como **emisión de paquetes de onda wavelet 2D ($\Psi(\omega, t)$)** y decodificar bloques completos de tokens en **un solo forward pass**.

> ⚠️ **Estado del proyecto:** Este repositorio contiene experimentos preliminares. Los resultados actuales demuestran la **mecánica** del enfoque (invertibilidad exacta de la transformada wavelet, vocoder ligero, memorización de corpus pequeños), pero **no** validan aún la generación de lenguaje generalizable. Los benchmarks actuales usan mayoritariamente corpus sintéticos/hardcodeados y baselines parcialmente extrapolados. Ver la sección [Resultados Reales](#-resultados-reales-y-limitaciones).

---

## 🏗️ El Pipeline SpecWave: Wave-In ➔ Wave-Out

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                       END-TO-END SPECTRAL WAVE PIPELINE (SPEC-WAVE)                             │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                                 │
 │  1. PROMPT (WAVE-IN)        2. RESONANT REASONING (FREQUENCY)    3. SYNTHESIS (WAVE-OUT)        │
 │                                                                                                 │
 │  [Prompt Tokens]            [Pure Frequency Domain Transfer]     [Full Response Block]          │
 │         │                               │                                 ▲                     │
 │         ▼                               ▼                                 │                     │
 │  [2D DWT Encoder]    ───►   [DeltaPhase / Spectral Core]  ───►   [2D IDWT Vocoder]              │
 │  (Generates Ψ_in)           (Maps Ψ_in ──► Ψ_out in S¹)          (Synthesizes N tokens O(1))    │
 │                                                                                                 │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Qué está implementado y verificado

### 1. Transformada Wavelet 2D Exacta (verificado)
- `spec_wave/wavelet.py`: Transformada de Haar 2D (DWT/IDWT) **matemáticamente correcta**.
- Verifica la conservación de energía de Parseval a precisión de máquina (error ~6e-08).
- La reconstrucción es una biyección isométrica exacta (error máximo ~4.77e-07).
- **Test:** `tests/test_core.py` (Test 1) — ✅ PASA.

### 2. Vocoder Espectral Paralelo (verificado como mecánica)
- `spec_wave/vocoder.py`: Reconstruye embeddings desde 4 subbandas wavelet en un solo paso.
- Puede **memorizar** corpus pequeños (~12 bloques de 64 tokens) con exact match 100% y PPL 1.0009.
- **Limitación:** No hay split train/test en este benchmark; el PPL es de memorización, no de generalización.
- **Test:** `tests/benchmark_vocoder_fineweb.py` — usa 2 strings hardcodeados (NO descarga FineWeb/WikiText).

### 3. Latencia del Vocoder (medida, pero sin datos reales)
- Un forward único del vocoder es rápido: ~7-28 ms para N=32-256.
- **Limitación:** Los speedups de 104x-155x se calculan con un baseline **extrapolado** (constante `13.0 ms/token`), no medido. El throughput de 13,589 tok/s se mide sobre **ruido aleatorio**, no sobre generación de texto real.
- **Test:** `tests/benchmark_gpu_wallclock.py`.

### 4. Adaptador sobre GPT-2 Real (parcialmente verificado)
- `examples/adapt_gpt2_specwave.py`: Carga GPT-2 real (124M) desde HuggingFace, congela el 100% de los pesos.
- Puede memorizar 3 pares de texto hardcodeados en ~2 minutos (exact match 100%).
- **Limitación:** El speedup de 12.27x se calcula con una constante inventada (`25.0 ms/token`), no midiendo GPT-2 autoregresivo.

### 5. Generalización en WikiText-2 Real (resultado negativo pero honesto)
- `examples/benchmark_ppl_parity.py`: Descarga WikiText-2 real, usa split train/test estricto.
- **Resultado:** Train PPL 1.02 (memorización) vs **Val PPL ~4,500** (no generaliza).
- Este es el experimento más honesto del repo y muestra la brecha real: el vocoder memoriza pero no generaliza.

---

## 📊 Resultados Reales y Limitaciones

| Claim del README original | Realidad verificada |
| :--- | :--- |
| "250x faster than autoregressive" | No medido. Los speedups de 104x-155x usan baselines extrapolados con constantes arbitrarias. |
| "100.00% lossless token recovery (PPL 1.0009)" | Cierto solo sobre corpus hardcodeados de ~12 bloques (memorización, sin split). |
| "TinyStories pre-training, 50.29x" | 8 plantillas sintéticas hardcodeadas, val = clon del train, baseline propio sin KV-cache. |
| "13,589.7 tokens/sec peak throughput" | Medido sobre vocoder sin entrenar con entrada aleatoria (ruido), no sobre texto. |
| "GPT-2 retrofitting, 12.27x-80x" | GPT-2 real cargado, pero 3 muestras hardcodeadas y baseline con constante inventada. |
| "Safety: 100% attack interception en 0.0937 ms" | Clasificador sobre ruido gaussiano con medias opuestas (+1.2 vs -1.2), trivialmente separable. No hay texto ni jailbreaks reales. |
| "PPL=1.02 convergence en T4" | Train PPL 1.02 (memorización). Val PPL ~4,500: **no generaliza**. |

---

## ⚡ Quickstart

```bash
# Clone and install
git clone https://github.com/mrcm-org/spec-wave.git
cd spec-wave
pip install -e .

# Run core test suite (wavelet invertibility + latency + synthetic pipeline)
python tests/test_core.py
```

### Output del test core (verificado):
```text
================================================================================
🌊 TEST 1: Exact 2D Haar Wavelet Inversion & Parseval Energy Conservation
================================================================================
Original Spatial Energy:      16158.885742
Wavelet Subband Energy:       16158.884766
Parseval Energy Error:        6.04e-08 (Exact Machine Precision)
Max Absolute 2D IDWT Error:   4.77e-07
✅ Result: 2D Wavelet representation is an exact isometric bijection (PASSED).

================================================================================
⚡ TEST 2: Single-Shot O(1) Generation Latency (N=64)
================================================================================
Single-Shot SpecWave O(1) Latency: 0.659 ms
✅ Result: Single-shot spectral waveform decoding achieves sub-millisecond generation.

================================================================================
🎯 TEST 3: End-to-End Pure Spectral Wave Pipeline (Wave-In -> Wave-Out)
================================================================================
Final End-to-End Recovery Accuracy: 100.00%
✅ Result: End-to-End Pure Spectral Wave Pipeline verified with 100% SUCCESS!
```

> ⚠️ **Nota sobre el Test 3:** La tarea "end-to-end" es aprender la función sintética `(prompt*3+7) % 256`. La red la memoriza en 200 pasos. No es generación de lenguaje real.

---

## 📚 Estructura del Repositorio

```text
spec-wave/
├── spec_wave/
│   ├── __init__.py      # Package exports
│   ├── wavelet.py       # 2D DWT & IDWT Exact Lossless Wavelet Operators
│   ├── vocoder.py       # Parallel Spectral Language Vocoder & Refiner
│   ├── model.py         # SpecWave Language Model Architecture
│   └── pipeline.py      # End-to-End Spectral Wave Pipeline (Wave-In -> Wave-Out)
├── docs/                # Informes de experimentos (reescritos para reflejar lo medido)
├── tests/
│   ├── test_core.py     # Test suite oficial (wavelet, latencia, pipeline sintético)
│   ├── benchmark_vocoder_fineweb.py   # Fase 1: corpus hardcodeado
│   ├── benchmark_gpu_wallclock.py     # Fase 3: latencia + baseline extrapolado
│   └── benchmark_spectral_safety.py   # Fase 4: ruido gaussiano sintético
├── examples/
│   ├── adapt_gpt2_specwave.py         # Fase 4A: GPT-2 real + 3 muestras
│   ├── benchmark_ppl_parity.py        # WikiText-2 real (el más honesto)
│   ├── benchmark_streaming_generalization.py  # WikiText-2 real, val PPL ~4,500
│   ├── train_tinystories_specwave.py  # Fase 2: 8 plantillas sintéticas
│   └── specwave_gpt2_colab_demo.ipynb # Demo Colab
├── setup.py             # Package Configuration
└── README.md            # Project Overview
```

---

## 🗺️ Roadmap de Validación Empírica

El roadmap original (`docs/empirical_validation_roadmap.md`) describe los experimentos que **habría que hacer** para validar la idea. Los scripts actuales **no cumplen** esos requisitos. Estado real:

| Fase | Objetivo del roadmap | Estado real | Script |
| :--- | :--- | :--- | :--- |
| **P1** | Vocoder en FineWeb (100k muestras) | ❌ Corpus hardcodeado (~12 bloques) | `tests/benchmark_vocoder_fineweb.py` |
| **P2** | TinyStories real (2.1M historias) | ❌ 8 plantillas sintéticas | `examples/train_tinystories_specwave.py` |
| **P3** | Benchmarks GPU (Triton/CUDA, 250x) | ⚠️ Latencia medida, baseline extrapolado | `tests/benchmark_gpu_wallclock.py` |
| **P4** | Safety con jailbreaks reales | ❌ Ruido gaussiano sintético | `tests/benchmark_spectral_safety.py` |
| **P4C** | WikiText-2 real, generalización | ⚠️ Datos reales, pero val PPL ~4,500 (no generaliza) | `examples/benchmark_ppl_parity.py` |

---

## 📄 Paper Draft

El borrador del paper (`docs/spec_wave_paper_draft.md`) contiene claims que **no están respaldados por los experimentos actuales** (p. ej., "100.00% lossless token recovery" como resultado general, "155.56x speedup" con baseline extrapolado). **No recomendamos enviarlo a revisión por pares en su estado actual.**

---

## 📜 Licencia

Distribuido bajo la **MIT License**. Ver `LICENSE`.