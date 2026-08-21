# 🌊 SpecWave: Exploración Espectral 2D y Speculative Decoding Multi-Token

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status: Audited & Verified](https://img.shields.io/badge/Status-Audited%20%26%20Verified-brightgreen.svg)](docs/auditoria_conjunta_2026-08-21.md)

**SpecWave** es un proyecto de investigación experimental que investiga el uso de transformadas wavelet 2D ($\Psi(\omega, t)$) para la síntesis de lenguaje no-autorregresiva (NAR) y la aceleración de modelos causales mediante **Speculative Decoding Multi-Token (MTP Drafter)**.

---

## 🎯 Veredicto Científico y Hallazgos Principales

Tras una rigurosa batería de experimentos a escala (hasta 200M tokens) y una **auditoría conjunta independiente** ([`docs/auditoria_conjunta_2026-08-21.md`](docs/auditoria_conjunta_2026-08-21.md)), el proyecto ha establecido conclusiones definitivas:

```
                                  MAPA DE RESULTADOS CIENTÍFICOS
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. TRANSFORMADA WAVELET 2D:   Biyección ortogonal exacta (Parseval a precisión de máquina).  │
 │ 2. DISPARO ÚNICO (ONE-SHOT):  Suelo entrópico informacional (Óptimo de Bayes PPL ~330-370). │
 │ 3. SPECULATIVE DECODING v2:   🏆 2.23x menos pases | 2.20 tok/paso | 1.17x SPEEDUP NETO CPU.│
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. El Suelo Entrópico de la Generación One-Shot
* La factorización no-autorregresiva pura $P(Y \mid X) \approx \prod_{i=1}^{L} P(y_i \mid X)$ colisiona con la suma de entropías marginales del lenguaje.
* Cinco arquitecturas distintas (MLP 21M, Transformer 110M, Deep GPT-2 124M, Native 34M y Diffusion DDIM) y un entrenamiento masivo de **200 Millones de Tokens** alcanzaron la misma meseta asintótica ($\text{PPL} \approx 330\text{--}370$ en TinyStories, $\text{PPL} \approx 530\text{--}1000$ en WikiText-2).

### 2. 🏆 La Vía Ganadora: Speculative Decoding con MTP Wavelet Drafter
* Reconversión del vocoder espectral en un **Drafter Multi-Token ligero de cero redundancia** ($\sim 3\text{M}$ params) sobre los estados cacheados del modelo causal (GPT-2).
* **Métricas Auditadas con KV-Cache Completo:**
  * **$2.23\times$ reducción en pases forward de GPT-2** ($583$ vs $1,300$).
  * **$2.20$ tokens generados por paso forward** (frente a 1.00 en AR).
  * **$\mathbf{1.17\times}$ Aceleración Real Neta (Wall-Clock Speedup)** en CPU frente a GPT-2 optimizado con KV-cache.
  * **Paridad Matemática Exacta:** Muestreo por rechazo riguroso que garantiza una distribución de salida idéntica a GPT-2 ($PPL = 10.73$).

---

## 📊 Matriz Comparativa de Experimentos

| Arquitectura / Experimento | Dataset & Tokens | Métrica Clave | Veredicto / Impacto |
| :--- | :--- | :---: | :--- |
| **Invertibilidad Vocoder (Fase 1)** | WikiText-2 Real | Acc 97.45% / PPL 1.46 | Biyección isométrica exacta |
| **Ablación Wavelet vs Plano (Fase 2)** | WikiText-2 Real | $\Delta \text{Loss} = -0.0001$ | Isomorfismo ortogonal ($B \approx A$) |
| **GPT-2 Adaptado (TinyStories)** | 1.92M tokens | Val PPL 307.68 | ~155 tok/s (pesado) |
| **Native SpecWave LM (From-Scratch)** | 7.68M tokens | Val PPL 337.45 | 2,561 tok/s ($\mathbf{16.5\times}$ más rápido en train) |
| **Native Massive Overnight** | **200M tokens** (20.2 h) | Val PPL 369.51 | Límite de capacidad paramétrica |
| **Ablación de Pérdidas (3 seeds)** | TinyStories (9 runs) | CE: **376.55** vs Híbrida: 378.03 | **CE Pura estadísticamente superior** |
| **Mask-Predict Iterativo (CMLM)** | 1.92M tokens | PPL 2,939 (R1 a R8) | Colapso de correlación condicional |
| **Speculative Decoding v2 (MTP)** | TinyStories + GPT-2 | **$2.23\times$ passes, $1.17\times$ speedup** | 🏆 **Ganador para servicio y producción** |

---

## ⚡ Quickstart & Reproducibilidad

### 1. Instalación
```bash
git clone https://github.com/mrcm-org/spec-wave.git
cd spec-wave
pip install -e .
pip install transformers tiktoken datasets
```

### 2. Test Suite Oficial (Smoke Latency & Parseval Conservation)
```bash
python tests/test_core.py
```

### 3. Ejecución del Benchmark de Speculative Decoding v2 (Auditado)
```bash
python -u examples/benchmark_wavelet_speculative_decoding_v2.py --num_prompts 20 --gen_length 64 --burst_len 4 --temperature 0.7
```

### 4. Ejecución del Estudio de Ablación de Pérdidas (3 Semillas)
```bash
python -u examples/ablation_loss_study.py --max_train_pairs 4000 --max_test_pairs 400 --batch_size 32
```

---

## 📚 Estructura del Repositorio

```text
spec-wave/
├── spec_wave/
│   ├── __init__.py           # Exportaciones del paquete
│   ├── wavelet.py            # Operadores DWT y IDWT 2D de Haar exactos (Parseval)
│   ├── vocoder.py            # Vocoder espectral y refinadores residuales 1D
│   ├── model.py              # Arquitectura base de modelos espectrales
│   ├── native_model.py       # Modelo NativeSpecWaveLM 100% from-scratch
│   └── pipeline.py           # Pipeline sintético Wave-In -> Wave-Out
├── docs/                     # Informes técnicos y auditorías independientes
│   ├── auditoria_conjunta_2026-08-21.md        # Síntesis de auditoría independiente
│   ├── findings_speculative_decoding_v2.md     # Benchmark riguroso de Speculative v2
│   ├── findings_loss_ablation_study.md         # Ablación CE vs Híbrida (3 seeds)
│   ├── findings_massive_200m_native_specwave.md# Run masivo de 200M tokens
│   ├── findings_native_specwave_from_scratch.md# Modelo nativo desde cero
│   └── empirical_validation_roadmap.md         # Hoja de ruta empírica
├── examples/                 # Scripts reproducibles de entrenamiento y evaluación
│   ├── benchmark_wavelet_speculative_decoding_v2.py # Speculative Drafter con KV-cache
│   ├── ablation_loss_study.py                  # Ablación de pérdidas multiescala
│   ├── train_native_specwave_decay.py          # Entrenamiento nativo con Horizon Decay
│   ├── train_iterative_mask_predict.py         # Refinamiento iterativo Mask-Predict
│   └── train_tinystories_streaming_specwave.py # Pipeline de streaming continuo
├── tests/
│   └── test_core.py          # Test suite (Parseval, latencia, recuperación sintética)
└── README.md
```

---

## 📜 Licencia

Distribuido bajo la **MIT License**.