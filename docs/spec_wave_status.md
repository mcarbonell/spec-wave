# SpecWave: Exploración de Síntesis de Lenguaje No-Autorregresiva vía Wavelets 2D — Borrador de Trabajo

**Mario Raúl Carbonell Martínez**  
*Investigación independiente*  
[https://github.com/mcarbonell/spec-wave](https://github.com/mcarbonell/spec-wave)  
Agosto 2026

> ⚠️ **Estado: Borrador de trabajo preliminar.** Este documento describe una hipótesis de investigación y los experimentos iniciales realizados. Los resultados actuales son **exploratorios** y no deben interpretarse como validación del enfoque. Los claims de speedup y "lossless recovery" del README original no están respaldados por los experimentos descritos aquí.

---

## Resumen

Los modelos de lenguaje autoregresivos generan texto token a token, requiriendo $N$ forward passes para producir $N$ tokens. Este trabajo explora una alternativa no-autorregresiva: formular la generación como **emisión de paquetes de onda wavelet 2D** y decodificar bloques completos mediante un **vocoder wavelet paralelo** en un solo forward pass.

Los experimentos iniciales demuestran:

1. **La transformada de Haar 2D es una biyección isométrica exacta** (conserva energía a precisión de máquina, error de reconstrucción < 5e-07).
2. **Un vocoder wavelet puede memorizar corpus pequeños**: reconstrucción exacta (100%) de ~12 bloques de 64 tokens hardcodeados (PPL 1.0009) y de 3 pares prompt→respuesta con GPT-2 real congelado.
3. **El vocoder es computacionalmente ligero**: latencia de ~7-28 ms para bloques de 32-256 tokens en un forward único.
4. **El enfoque NO generaliza aún**: en WikiText-2 real (600 pares de entrenamiento, 100 de validación blind), el train PPL converge a 1.02 pero la validación se estanca en PPL ~4,500.

**Conclusión del borrador:** La mecánica del enfoque (wavelets 2D + vocoder paralelo) funciona, pero la generación de lenguaje generalizable **no está demostrada**. Se requiere investigación adicional sustancial.

---

## 1. Introducción

La generación autoregresiva domina el modelado de lenguaje:
$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{N} P(y_t \mid y_{<t}, \mathbf{x})$$

Esto impone una dependencia secuencial: $N$ forward passes para $N$ tokens. La comunidad de audio resolvió un problema análogo (WaveNet → Mel-spectrograms + vocoders paralelos como HiFi-GAN). Este trabajo explora si una analogía similar es viable para lenguaje: representar la salida como una "onda de pensamiento" continua y decodificarla con un vocoder paralelo.

**Hipótesis:** Si los embeddings de tokens forman un manifold continuo, una transformada wavelet 2D puede representar bloques completos de texto, y un vocoder entrenado puede invertir esa representación en tokens.

---

## 2. Formulación Matemática y Arquitectura

### 2.1 Análisis Wavelet 2D (Wave-In)

Sea $\mathbf{X} \in \mathbb{R}^{N \times d_{\text{model}}}$ la matriz de embeddings de una secuencia de tokens. La transformada de Haar 2D la descompone en 4 subbandas ortogonales:
$$\mathbf{X} \xrightarrow{\text{2D DWT}} \{\mathbf{LL}, \mathbf{LH}, \mathbf{HL}, \mathbf{HH}\} \in \mathbb{R}^{\frac{N}{2} \times \frac{d_{\text{model}}}{2}}$$

Por el teorema de Parseval:
$$\|\mathbf{X}\|_F^2 = \|\mathbf{LL}\|_F^2 + \|\mathbf{LH}\|_F^2 + \|\mathbf{HL}\|_F^2 + \|\mathbf{HH}\|_F^2$$

**Verificado empíricamente:** error de Parseval ~6e-08, error máximo de reconstrucción ~4.77e-07 (precisión de máquina).

### 2.2 Razonador en Dominio de Frecuencia

El razonador mapea el espectro de entrada al espectro de salida:
$$\boldsymbol{\Psi}_{\text{out}} = \mathcal{F}_{\boldsymbol{\Theta}}(\boldsymbol{\Psi}_{\text{in}})$$

En la implementación actual, $\mathcal{F}_{\boldsymbol{\Theta}}$ es un MLP de 2-3 capas densas. **Nota:** No hay compresión — la dimensionalidad del espectro de salida es igual a la de entrada ($N \times d$).

### 2.3 Vocoder Espectral Paralelo (Wave-Out)

El vocoder invierte el espectro en embeddings continuos en un solo paso:
$$\hat{\mathbf{E}} = \text{2D IDWT}(\mathbf{LL}_{\text{out}}, \mathbf{LH}_{\text{out}}, \mathbf{HL}_{\text{out}}, \mathbf{HH}_{\text{out}})$$

Seguido de refinamiento convolucional residual y proyección al vocabulario:
$$\mathbf{Z} = \tilde{\mathbf{E}} \mathbf{W}_{\text{vocab}}^\top \in \mathbb{R}^{N \times |\mathcal{V}|}$$

---

## 3. Resultados Empíricos (Estado Actual)

### 3.1 Invertibilidad del Vocoder (Corpus Hardcodeado)

**Protocolo real:** 2 strings literales (texto sobre relatividad + código Python) embebidos en `tests/benchmark_vocoder_fineweb.py`, tokenizados con GPT-2 BPE, ~12 bloques de 64 tokens con solapamiento y duplicados. **Sin split train/test.**

| Métrica | Resultado |
| :--- | :---: |
| Exact Token Match | 100.00% |
| Reconstruction PPL | 1.0009 |
| Tiempo de entrenamiento | 49.33 s (CPU) |

**Interpretación:** Memorización de un corpus pequeño. No demuestra generalización.

### 3.2 Pre-Training en 8 Historias Sintéticas

**Protocolo real:** 8 plantillas hardcodeadas (NO TinyStories real), val = clon del train, baseline = mini-transformer propio sin KV-cache.

| Métrica | Baseline Causal | SpecWave |
| :--- | :---: | :---: |
| Latencia (32 tokens) | 415.90 ms | 8.27 ms |
| Exact Match | 100.00% | 100.00% |

**Interpretación:** El speedup de 50.29x es real pero sobre un baseline desventajado (sin KV-cache). El 100% es memorización de 8 plantillas.

### 3.3 Latencia y Throughput

**Protocolo real:** Latencia del vocoder medida sobre entrada aleatoria. Baseline extrapolado con fórmula `n * 13.0 + n² * 0.015` para N≥128 (constante arbitraria). Throughput sobre ruido aleatorio.

| N | SpecWave (medido) | Baseline (método) | Speedup |
| :--- | :---: | :---: | :---: |
| 32 | 7.33 ms | 272.16 ms (medido) | 37.14x |
| 64 | 10.19 ms | 630.88 ms (medido) | 61.90x |
| 128 | 18.20 ms | 1,909.76 ms (**extrapolado**) | 104.95x |
| 256 | 27.71 ms | 4,311.04 ms (**extrapolado**) | 155.56x |

**Interpretación:** La latencia de SpecWave es real y baja. Los speedups de 104x-155x **no son mediciones** — dependen de una constante arbitraria.

### 3.4 Retrofitting de GPT-2 Real (124M)

**Protocolo real:** GPT-2 real de HuggingFace, 100% congelado, 3 pares de texto hardcodeados, baseline autoregresivo **no medido** (constante `25.0 ms/token`).

| Métrica | Valor |
| :--- | :---: |
| Exact Match (3 pares) | 100.00% |
| Latencia SpecWave (N=64, CPU) | 130.40 ms |
| Latencia GPT-2 AR (N=64, CPU) | 1,600.00 ms (**constante, no medida**) |
| Speedup | 12.27x (derivado de la constante) |

### 3.5 Generalización en WikiText-2 Real (Resultado Clave)

**Protocolo real:** WikiText-2 descargado de HuggingFace, 600 pares train, 100 pares test blind, GPT-2 real congelado (capa 11 descongelada), weight-tying del LM head.

| Métrica | Train | Val (blind) |
| :--- | :---: | :---: |
| Loss final | 0.02 | ~8.4 |
| PPL final | 1.02 | ~4,500 |

**Interpretación:** El modelo memoriza el train (PPL 1.02) pero **no generaliza** (val PPL ~4,500). Un GPT-2 nativo tiene PPL ~25-35 en WikiText-2. La brecha de 4 órdenes de magnitud entre train y val es overfitting severo.

---

## 4. Limitaciones y Trabajo Futuro

### Limitaciones identificadas

1. **Sin generalización demostrada:** Todos los resultados de "100.00% exact match" son memorización de corpus pequeños (12 bloques, 8 historias, 3 pares, 600 pares).
2. **Speedups no verificados:** Los claims de 104x-155x usan baselines extrapolados con constantes arbitrarias. El baseline real (GPT-2 con KV-cache) nunca se mide.
3. **Sin compresión:** El espectro de salida tiene la misma dimensionalidad que la salida. El razonador MLP aprende un mapeo $N \times d \to N \times d$, lo que escala mal.
4. **Sin decodificación:** Solo greedy argmax. Sin beam search, sampling o verificación.
5. **Longitud fija:** La generación es de longitud fija ($N$), sin parada dinámica.
6. **El "LL subband = intención semántica" no está fundamentado:** La energía de las subbandas de Haar depende de diferencias entre filas/columnas adyacentes de la matriz de embeddings, no del significado.

### Trabajo futuro necesario

1. **Dataset real grande:** FineWeb o TinyStories real con split train/test estricto (N > 50,000 muestras).
2. **Baseline real:** Medir GPT-2 con KV-cache en el mismo hardware y con los mismos datos.
3. **Ablación:** Reemplazar DWT/IDWT por un MLP plano para saber si las wavelets aportan algo.
4. **Métricas de generalización:** PPL de validación como métrica principal, no exact match de train.
5. **Decodificación:** Añadir sampling/beam search y parada dinámica.

---

## 5. Conclusión

SpecWave explora una idea legítima (generación NAR vía vocoder wavelet paralelo) con una implementación limpia y una transformada wavelet matemáticamente correcta. Sin embargo, el estado actual **no valida** la generación de lenguaje generalizable: los experimentos demuestran memorización de corpus pequeños y latencia baja del vocoder, pero la generalización a texto no visto falla (val PPL ~4,500 en WikiText-2). Se requiere investigación adicional sustancial antes de que este enfoque pueda considerarse una alternativa viable a la generación autoregresiva.

---

## Referencias

1. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.
2. Brown, T., et al. (2020). *Language models are few-shot learners*. NeurIPS.
3. Oord, A., et al. (2016). *WaveNet: A generative model for raw audio*. arXiv:1609.03499.
4. Kong, J., et al. (2020). *HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis*. NeurIPS.
5. Eldan, R., & Li, Y. (2023). *TinyStories: How small can language models be and still speak coherent English?* arXiv:2305.07759.