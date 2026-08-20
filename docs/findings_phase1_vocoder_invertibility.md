# 🔬 Phase 1 Benchmark Report: Vocoder Invertibility at Scale (WikiText-2 Real)

> **STATUS: [VALIDADO / PUERTA 1 PASADA / PPL TEST 1.4568 / EXACTITUD 97.45%]**  
> Validación empírica de la capacidad de inversión/autoencoding del **Parallel 2D Wavelet Spectral Language Vocoder** sobre **WikiText-2 real**, tokenizado con el tokenizador GPT-2 BPE ($V = 50,257$).  
> **Script reproducible:** [`benchmarks/phase1_vocoder_scale.py`](../benchmarks/phase1_vocoder_scale.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Medido (Full Test Split) | Umbral Puerta 1 | Estado |
| :--- | :---: | :---: | :---: |
| **Dataset** | **WikiText-2 real** (10,000 bloques train / 1,000 bloques test) | Datos reales HF/PyTorch | ✅ Validado |
| **Reconstruction Test PPL** | **1.4568** | $\le 2.0$ | **✅ PASA** |
| **Test Token Accuracy (%)** | **97.45%** | $\ge 95.0\%$ | **✅ PASA** |
| **Test Exact Sequence Match** | **23.50%** | — | Bloques completos (64 tokens) idénticos |
| **Cross-Entropy Loss (Test)** | **0.3762** | — | — |
| **Vocab Scale** | **50,257 tokens (GPT-2 BPE)** | Tokenizador real | ✅ Validado |
| **Tiempo de Entrenamiento** | **508.14 s** (~8.4 min en CPU Zen 4) | — | 0.93 steps/s |

```
                       TRAYECTORIA DE RECONSTRUCCIÓN EN TEST CIEGO (FASE 1)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Step   1 (Init):  Loss = 10.1790 │ Val PPL = 26,344.54 │ Val Tok Acc =  2.52%               │
 │ Step 100:         Loss =  0.8260 │ Val PPL =      2.28 │ Val Tok Acc = 93.16%               │
 │ Step 200:         Loss =  0.4683 │ Val PPL =      1.60 │ Val Tok Acc = 96.30%               │
 │ Step 300:         Loss =  0.3942 │ Val PPL =      1.48 │ Val Tok Acc = 97.35%               │
 │ Step 400:         Loss =  0.3782 │ Val PPL =      1.46 │ Val Tok Acc = 97.45%               │
 │ Final (Full Test):Loss =  0.3762 │ Val PPL =      1.46 │ Val Tok Acc = 97.45% ──► GATE 1 PASS│
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental Real

### 2.1 Dataset y Particiones
- **Fuente:** WikiText-2 real (2.45M tokens de entrenamiento, 295k tokens de test).
- **Partición Train:** 10,000 bloques contiguos de 64 tokens (640,000 tokens procesados).
- **Partición Test Blind:** 1,000 bloques independientes de 64 tokens (64,000 tokens ciegos evaluados).
- **Tokenizador:** GPT-2 BPE oficial ($V=50,257$).

### 2.2 Arquitectura del Modelo
```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                           ARQUITECTURA EXPERIMENTAL (FASE 1)                                │
 ├───────────────────────────────┬─────────────────────────────────────────────────────────────┤
 │           LAYER               │                       SPECIFICATION                         │
 ├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ 1. Token Embeddings           │ Embedding(num_embeddings=50257, embedding_dim=128)          │
 │ 2. 2D Haar DWT Analysis       │ Matrix-free downsampling -> 4 subbands [B, 32, 64]          │
 │ 3. 2D Haar IDWT Synthesis     │ Matrix-free parallel upsampling -> [B, 64, 128]             │
 │ 4. Conv1D Spectral Refiner    │ Conv1d(128, 256, k=3, p=1) + GELU + Conv1d(256, 128, k=3)  │
 │ 5. Parallel De-quantizer Head │ Linear(in_features=128, out_features=50257, bias=False)    │
 └───────────────────────────────┴─────────────────────────────────────────────────────────────┘
 Total de Parámetros: 13,063,040
```

### 2.3 Hiperparámetros
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=468, eta_min=1e-5)
batch_size = 64 (4,096 tokens/paso)
epochs = 3
seq_len = 64
d_model = 128
```

---

## 🔬 3. Dinámica de Entrenamiento Detallada

| Época | Paso | Train Loss | Train PPL | Blind Val Loss | Blind Val PPL | Val Token Acc (%) | Val Exact Seq Match (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 10.9579 | 57,405.44 | 10.1790 | 26,344.54 | 2.52% | 0.00% |
| **1** | 100 | 0.6452 | 1.91 | 0.8260 | 2.28 | 93.16% | 2.50% |
| **2** | 200 | 0.1380 | 1.15 | 0.4683 | 1.60 | 96.30% | 13.30% |
| **2** | 300 | 0.0907 | 1.09 | 0.3942 | 1.48 | 97.35% | 22.30% |
| **3** | 400 | 0.0488 | 1.05 | 0.3782 | 1.46 | 97.45% | 23.50% |
| **Final** | **468** | **0.0312** | **1.03** | **0.3762** | **1.4568** | **97.45%** | **23.50%** |

---

## 💡 4. Interpretación de los Resultados

1. **El Vocoder como Autoencoder generaliza:** A diferencia del generador autorregresivo/NAR completo, el pipeline del vocoder (Embeddings $\to$ DWT 2D $\to$ IDWT 2D $\to$ Refiner $\to$ LM Head) es capaz de auto-reconstruir texto real nunca visto con una exactitud por token del **97.45%** y una perplejidad en test ciego de **1.4568** (superando holgadamente el criterio de la puerta de $PPL \le 2.0$).
2. **Capacidad de Inversión Confirmada:** Esto demuestra empíricamente que una vez obtenido un manifold continuo adecuado de embeddings vía IDWT 2D, el vocoder puede proyectar y des-cuantizar los tokens de vuelta al vocabulario con pérdidas mínimas.
3. **Paso a la Fase 2:** Con la invertibilidad del vocoder validada a escala, la siguiente pregunta crítica es: ¿Aportan las wavelets 2D una ventaja real frente a un procesador plano (Ablación Wavelet vs. Flat)?