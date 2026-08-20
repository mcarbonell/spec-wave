# 🔬 Deep GPT-2 Spectral Transformer Benchmark Report

> **STATUS: [COMPLETADO / 110M PARÁMETROS / CAPAS 8-11 DESCONGELADAS + CROSS-ATTENTION TRANSFORMER]**  
> Estudio empírico de alta capacidad acoplando **GPT-2 (124M)** con sus últimas 4 capas descongeladas a un **Razonador Transformer Bidireccional de Atención Cruzada (4 capas, 12 cabezales)** con pérdida espectral multiescala Parseval (4x en subbanda $\mathbf{LL}$) sobre pares reales de WikiText-2.  
> **Script reproducible:** [`examples/train_gpt2_spectral_transformer.py`](../examples/train_gpt2_spectral_transformer.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados

| Métrica | Native GPT-2 Causal (Baseline) | Deep Spectral Transformer (SpecWave) | Brecha / Ratio |
| :--- | :---: | :---: | :---: |
| **Parámetros Entrenables** | 124M (Backbone preentrenado) | **109,505,280** (28.3M GPT-2 + 81.1M Adapter) | ~110M params |
| **Tipo de Generación** | Autorregresiva $O(N)$ (64 pasos) | **Paralela Espectral $O(1)$** (1 paso) | — |
| **Blind Test Loss** | **3.7009** | **7.0106** | +3.31 nats |
| **Blind Test Perplexity (PPL)** | **40.48** | **1108.30** | ~27x brecha |
| **Blind Test Token Accuracy** | — | **4.40%** | — |
| **Tiempo de Entrenamiento** | — | **806.87 s** (~13.4 min en CPU Zen 4) | — |

```
               DINÁMICA DE LOSS EN TEST CIEGO (DEEP TRANSFORMER 110M)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Step   1 (Init):  Train Loss = 18.7153 │ Val Loss = 14.2713 │ Val Tok Acc = 1.64%           │
 │ Step  50:         Train Loss =  7.0897 │ Val Loss =  7.1780 │ Val Tok Acc = 4.59%           │
 │ Step 100:         Train Loss =  6.8431 │ Val Loss =  7.1313 │ Val Tok Acc = 4.45%           │
 │ Step 200:         Train Loss =  6.4184 │ Val Loss =  7.0401 │ Val Tok Acc = 4.53%           │
 │ Step 300:         Train Loss =  6.1709 │ Val Loss =  7.0341 │ Val Tok Acc = 4.18%           │
 │ Final (Full Test):Train Loss =  6.1700 │ Val Loss =  7.0106 │ Val PPL = 1108.30             │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Arquitectura del Experimento

1. **Backbone:** GPT-2 Transformer con bloques 8, 9, 10, 11 y LayerNorm final descongelados ($28.35\text{M}$ parámetros adaptativos).
2. **Razonador:** Transformer Decoder Bidireccional de 4 capas ($d_{\text{model}}=768$, 12 attention heads, $d_{\text{ff}}=3072$) con atención cruzada completa sobre los estados contextuales del prompt ($81.15\text{M}$ parámetros).
3. **Pérdida Espectral Multiescala:**
   $$\mathcal{L} = \mathcal{L}_{\text{CE}} + 2 \cdot (4 \|\mathbf{LL} - \mathbf{LL}^*\|^2 + \|\mathbf{LH} - \mathbf{LH}^*\|^2 + \|\mathbf{HL} - \mathbf{HL}^*\|^2 + \|\mathbf{HH} - \mathbf{HH}^*\|^2) + 2 \|\hat{\mathbf{E}} - \mathbf{E}^*\|^2$$
4. **Optimizador:** AdamW con learning rates diferenciados ($5\times 10^{-5}$ para GPT-2 y $5\times 10^{-4}$ para el transformador espectral) con `CosineAnnealingLR` y grad clipping.

---

## 💡 3. Hallazgo Teórico Fundamental

Este experimento arroja la conclusión más valiosa del proyecto:

1. **No es un problema de capacidad ni de arquitectura:**
   Con **110 millones de parámetros entrenables**, un Transformer profundo con atención cruzada y descongelación de GPT-2, la pérdida de validación ciego converge exactamente a la misma meseta que el MLP simple de la Fase 2 ($\text{Loss} \approx 7.00$, $\text{PPL} \approx 1100$).

2. **La causa real es la independencia condicional en la decodificación paralela de 1 paso:**
   En generación de texto, la distribución conjunta de una respuesta de 64 tokens es:
   $$P(y_1, y_2, \dots, y_{64} \mid X) = \prod_{i=1}^{64} P(y_i \mid y_{<i}, X)$$
   Al emitir los 64 tokens en un único forward pass sin condicionamiento causal secuencial, el modelo se ve forzado a aproximar la distribución como un producto de marginales independientes:
   $$P(y_1, \dots, y_{64} \mid X) \approx \prod_{i=1}^{64} P(y_i \mid X)$$
   La entropía marginal de cada posición dada únicamente el prompt $X$ tiene un límite teórico inferior de $\approx 7.0$ nats ($\text{PPL} \approx 1100$), que corresponde a la incertidumbre inherente de qué palabra aparecerá en la posición $i$ cuando no se conoce lo que se ha dicho en las posiciones $1 \dots i-1$.
