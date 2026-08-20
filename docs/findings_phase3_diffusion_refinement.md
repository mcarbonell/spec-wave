# 🔬 Phase 3 Benchmark Report: Spectral Wavelet Diffusion-LM (Iterative Refinement)

> **STATUS: [COMPLETADO / EVALUACIÓN DE DIFUSIÓN ESPECTRAL CONTINUA / GATE 3 EVALUADO]**  
> Evaluación empírica de la vía no-autorregresiva (NAR) iterativa mediante **modelado de difusión continuo (Diffusion-LM)** en el espacio de subbandas Wavelet 2D acoplado al Vocoder IDWT 2D paralelo, sobre datos reales de WikiText-2.  
> **Script reproducible:** [`benchmarks/phase3_diffusion.py`](../benchmarks/phase3_diffusion.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Medido | Interpretación |
| :--- | :---: | :--- |
| **Dataset** | **WikiText-2 real** (2,000 pares train, 200 pares blind test) | Secuencias contiguas reales |
| **Espacio de Difusión** | Continuous 2D Wavelet Spectrum ($\mathbb{R}^{B \times 8192}$) | $4 \times 32 \times 64 = 8,192$ dims |
| **Diffusion Noise MSE (Final)** | **1.0006** | Predicción del ruido gaussiano condicional |
| **Sampling Method** | **10-step DDIM Inversion** | Denoising determinista rápido |
| **Final Blind Test Gen Accuracy** | **1.30%** | Generación de bloque condicional ciego |
| **Parámetros del Sistema** | **23,447,296** | Denoiser + Embeddings + Vocoder |
| **Tiempo de Entrenamiento** | **133.82 s** (CPU Zen 4) | ~2.2 minutos |

```
                 TRAYECTORIA DE DIFUSIÓN ESPECTRAL (FASE 3)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Step   1 (Init):  Total Loss = 2.4254 │ Diff MSE = 1.3359 │ Val Gen Acc (10 DDIM) = 0.00%   │
 │ Step  50:         Total Loss = 1.2848 │ Diff MSE = 1.0158 │ Val Gen Acc (10 DDIM) = 1.80%   │
 │ Step 100:         Total Loss = 1.1310 │ Diff MSE = 1.0094 │ Val Gen Acc (10 DDIM) = 1.24%   │
 │ Step 150:         Total Loss = 1.0980 │ Diff MSE = 1.0060 │ Val Gen Acc (10 DDIM) = 1.20%   │
 │ Final (Full Test):Diff MSE   = 1.0006 │ Val Gen Acc (10 DDIM) = 1.30%                      │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental

### 2.1 Arquitectura del Difusor Espectral
1. **Representación Objetivo ($\mathbf{z}_0$):** Target Tokens (64) $\to$ Embeddings ($64 \times 128$) $\to$ 2D Haar DWT $\to$ Espectro continuo $\mathbf{z}_0 \in \mathbb{R}^{B \times 8192}$.
2. **Proceso Forward de Difusión:** Cosine beta schedule ($T=50$ timesteps):
   $$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
3. **Denoiser de Ruido ($\boldsymbol{\epsilon}_\theta$):** Red residual con embeddings sinusoidales de tiempo y proyección contextual del prompt.
4. **Muestreo en Inferencia (DDIM 10 pasos):** Inversión iterativa desde ruido puro $\mathbf{z}_T \sim \mathcal{N}(0, \mathbf{I})$ hasta $\hat{\mathbf{z}}_0$.
5. **Síntesis Paralela (Vocoder):** $\hat{\mathbf{z}}_0 \xrightarrow{\text{2D IDWT}} \hat{\mathbf{E}} \xrightarrow{\text{Refiner}} \text{LM Head} \to \text{Tokens}$.

---

## 💡 3. Conclusiones Científicas y Veredicto de la Fase 3

1. **La Maldición de la Dimensionalidad en Bloques Completos:** Difundir un bloque continuo completo de $64 \times 128 = 8,192$ dimensiones de forma no secuencial presenta un desafío severo de optimización sin una red de atención profunda de cientos de millones de parámetros. El MSE del ruido converge a $\approx 1.00$, indicando que el denoiser predice aproximadamente la media de la distribución gaussiana sin resolver la estructura fina del texto.
2. **Comparativa vs. Disparo Único ($O(1)$):** Mientras que el autoencoder (Fase 1) es casi perfecto ($97.45\%$ exactitud), la generación de texto nuevo a partir de un prompt (sea en 1 paso con MLP o en 10 pasos con difusor ligero) sufre de una brecha fundamental frente a los modelos causales autorregresivos.
3. **Veredicto del Roadmap:** El plan de falsación temprana ha cumplido su objetivo de investigar rigurosamente los límites del enfoque espectral.
