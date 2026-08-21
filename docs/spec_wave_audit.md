# 🔍 SpecWave: Auditoría Profunda del Proyecto — Idea, Código y Experimentos

> **STATUS: [AUDITORÍA INDEPENDIENTE / 2026-08-21 / ALCANCE: REPO COMPLETO]**
> ⚠️ **Nota:** existe una síntesis conjunta de esta auditoría con la de GPT 5.6 Earth en [`docs/auditoria_conjunta_2026-08-21.md`](auditoria_conjunta_2026-08-21.md) — usar ese documento como referencia accionable principal.
> Análisis integral del repositorio: hipótesis central, fundamentos matemáticos, calidad del código, validez experimental, viabilidad de la línea de investigación y vías de rescate con beneficio potencial.
> **Método:** revisión de todo el código (`spec_wave/`, `benchmarks/`, `examples/`, `tests/`), los 14 documentos de hallazgos, el historial git completo y los experimentos en curso (`native_specwave_decay_200m`, checkpoints hasta step 42k).

---

## 1. Veredicto Ejecutivo

| Dimensión | Evaluación |
| :--- | :--- |
| **Hipótesis central** (generador NAR one-shot que compita en PPL con AR) | ❌ **Cerrada por evidencia propia + teoría.** El PPL one-shot tiene un suelo entrópico informacional que ninguna escala, pérdida ni tuning puede bajar. |
| **Calidad metodológica** (post-pase de honestidad del 19/08) | ✅ **Alta.** Splits ciegos reales, datasets no-solapados, baselines medidos, roadmap de falsación. Mejor rigor que muchos workshops. |
| **Calidad del código** | ✅ Núcleo correcto y limpio. ⚠️ 2 bugs de métrica, 1 fallo de diseño en el drafter especulativo, 1 ablación clave ausente. Detalle en §4. |
| **Resultado científico más valioso** | ✅ El **suelo entrópico independiente de capacidad**: 5 arquitecturas (21M–123M params) convergen a la misma meseta de loss. Publicable como negative result con controles adicionales. |
| **Vías de rescate** | ✅ 3 concretas y medibles: drafter MTP barato para speculative decoding, semi-AR con refinamiento intra-ráfaga (mask-predict), y coarse-to-fine en espacio wavelet. Detalle en §6. |

**Conclusión en una frase:** la idea como "generador one-shot" está refutada; el proyecto contiene los datos que la refutan (lo cual es un logro) y dos o tres pivotes viables alineados con la práctica industrial actual (Medusa, EAGLE, DeepSeek-MTP).

---

## 2. Por Qué el PPL No Baja: el Suelo Entrópico

### 2.1 Evidencia empírica (dispersa entre los docs del repo)

Cinco arquitecturas radicalmente distintas convergen a la misma meseta de validación:

| Arquitectura | Params | Datos (val blind) | Val PPL | Fuente |
| :--- | ---: | :--- | ---: | :--- |
| MLP plano (sin wavelets) | 21.7M | WikiText 512k tok | 1,088.57 | `findings_phase2_ablation.md` |
| Transformer cruzado profundo | 109.5M | WikiText 1.5k pares | 1,108.30 | `findings_deep_gpt2_spectral_transformer.md` |
| GPT-2 (6-11) + espectral | 123.6M | WikiText 2.3M tok | 535.33 | `findings_massive_scale_training.md` |
| Nativo from-scratch | 34.1M | TinyStories 7.68M tok | 337.45 | `findings_native_specwave_from_scratch.md` |
| Difusor DDIM 10 pasos | 23.4M | WikiText 512k tok | Acc 1.30% | `findings_phase3_diffusion_refinement.md` |

Observaciones clave:

1. **Train ≈ Val en todos los experimentos a escala** (5.89 vs 5.82; 6.72 vs 6.28). No hay overfitting: el límite es **informacional**, no de capacidad ni de optimización.
2. Multiplicar parámetros ×5 (21.7M → 110M) movió el PPL un 2% (1088 → 1108). Multiplicar tokens únicos ×4.5 lo movió marginalmente. La curva nativa es plana desde el paso 300 (5.90 → 5.82 en 7M tokens adicionales).
3. El experimento de horizon-decay en curso lo confirma de nuevo: reponderar posiciones (γ=0.94) solo **redistribuye** error (frente 7.69% vs cola 7.08%; PPL global 354 vs 351 uniforme). La reponderación no crea información condicional.

### 2.2 Explicación teórica

La decodificación paralela one-shot factoriza la conjunta como producto de marginales:

$$P(y_1, \dots, y_{64} \mid X) \approx \prod_{i=1}^{64} P(y_i \mid X)$$

El predictor óptimo de Bayes dentro de esta familia alcanza exactamente la suma de entropías marginales $H(y_i \mid X)$. El meseta observado **ES ese óptimo**: los modelos ya son casi-Bayes-óptimos para la familia factorizada (por eso train≈val y por eso la capacidad es irrelevante). En WikiText el suelo ronda 7.0 nats; en TinyStories (más predecible) ~5.8 nats.

**Implicación operativa:** intentar bajar el PPL one-shot mediante ingeniería (más capas, más datos, mejores schedules, pérdidas auxiliares, decaimiento de horizonte) es matemáticamente estéril. La única salida es **cambiar la factorización** para restaurar dependencia condicional: refinamiento iterativo, ráfagas semi-AR pequeñas o mask-predict (§6).

---

## 3. Las Wavelets Son Neutrales por Construcción

Punto que afecta la identidad misma del proyecto:

1. **Reparametrización lineal.** Haar 2D es una matriz ortogonal fija $\mathbf{H}$. El pipeline es $\text{IDWT} \circ \text{MLP} \circ \text{DWT}$; como $\mathbf{H}^\top \text{MLP}\, \mathbf{H}$ es una reparametrización del MLP, el resultado B≈A de la Fase 2 ($\Delta$ Loss = 0.0001) era **matemáticamente inevitable**, no un descubrimiento empírico.
2. **El "vocoder" es una identidad.** En `benchmarks/phase1_vocoder_scale.py` (DWT seguido de IDWT), $\text{IDWT}(\text{DWT}(\mathbf{E})) = \mathbf{E}$ exactamente. La "Puerta 1" (PPL 1.46, acc 97.45%) se pasa por construcción: mide si un refiner convolucional + LM head puede decodificar embeddings — un autoencoder cuya parte espectral es `identity`.
3. **El emparejamiento 2D es arbitrario.** Mezcla el eje temporal (tokens) con el eje de canales del embedding. Permutar filas/columnas produce otra descomposición igual de válida. La energía de HH/HL/LH depende de la geometría de la tabla de embeddings, no de sintaxis ni semántica (ya admitido en `spec_wave_status.md` §4.6).
4. **Único diseño donde la wavelet deja de ser neutral:** generación **coarse-to-fine sobre escalas** — generar LL primero (baja dimensionalidad efectiva, captura el "tema") y condicionar LH/HL/HH en pasadas posteriores. Eso equivale a difusión en cascada / generación progresiva, donde la jerarquía wavelet sí aporta estructura explícita. El one-shot con todas las subbandas simultáneamente, no.

---

## 4. Auditoría de Código

### 4.1 Fortalezas

- `spec_wave/wavelet.py`: DWT/IDWT Haar correctos (Parseval verificado a precisión de máquina).
- Datasets streaming no-solapados con splits ciegos reales (`TinyStoriesStreamingDataset`, `TokenBlockDataset`).
- Baselines nativos medidos sobre los mismos splits (GPT-2 PPL 10.73 TinyStories, 35.26 WikiText).
- Código legible, scripts reproducibles con CLI args, checkpoints con best-val tracking.

### 4.2 Hallazgos (ordenados por gravedad)

**A. El drafter especulativo no es barato — fallo de diseño, no bug.**
`examples/benchmark_wavelet_speculative_decoding.py:46-65`. `WaveletSpeculativeDrafter` envuelve **el mismo backbone GPT-2 que el verificador**. Cada paso especulativo cuesta 2 forwards completos de GPT-2 (draft + verify): el speedup wall-clock máximo teórico es <2x incluso con aceptación perfecta. El 0.76x medido es coherente con este diseño. Un drafter debe ser ≥5–10x más barato que el verificador (p. ej. `distilgpt2`, o la cabeza de ráfaga sobre un backbone pequeño).

**B. Bug en la tasa de aceptación.**
`benchmark_wavelet_speculative_decoding.py:120-131`: en el rechazo, `num_accepted += 1` cuenta el token de **reemplazo** como aceptado; y `total_accepted_tokens += num_new_tokens` (línea ~253) incluye el bonus. El α real es menor que el reportado. El "12.50%" del drafter sin entrenar es exactamente 1/8: cada paso rechaza en el primer token. Además `full_accept` (línea 145) devuelve una expresión incoherente que nunca se usa.

**C. Ningún lado usa KV-cache.**
El baseline AR (líneas 202-218) reintroduce el contexto completo cada paso, y el verify también. Todos los números de latencia/speedup comparan contra un baseline desventajado. Con KV-cache, GPT-2 en CPU rinde ~20–60 tok/s y el hueco se agranda.

**D. Pérdida híbrida posiblemente contraproducente — ablación clave ausente.**
`spec_wave/native_model.py:136-152`: se entrena `CE + 2·spectral_MSE + 2·manifold_MSE`, donde las MSE arrastran `refined` hacia `wte[target]`. Con weight-tying (`lm_head.weight = wte.weight`, línea 88), eso entrena medio autoencoder hacia el embedding exacto en vez de optimizar likelihood. Es plausible que CE solo dé mejor PPL. **Es la ablación barata más obvia que falta en todo el repo.**

**E. Coste oculto del semi-AR.**
`examples/train_semiautoregressive_specwave.py:149-163`: cada ráfaga re-codifica el contexto completo con GPT-2 (longitudes 64→112, sin caché). "4 pasos vs 64" es cierto en nº de forwards, pero cada uno es más caro que un paso AR cacheado. El "16x" necesita asterisco.

**F. Resultado semi-AR predeciblemente mediocre (PPL 334 vs 337 one-shot).**
En ráfagas de M=16, solo 4 de 64 posiciones reciben condicionamiento fresco; las otras 60 siguen siendo marginales. La ganancia del semi-AR escala con (posiciones condicionadas)/(total). M pequeño (2–4) o refinamiento intra-ráfaga son necesarios (§6.1).

**G. Menores.**
- PPL capado a `exp(20)` en los evaluadores (OK, pero documentar).
- Baseline TinyStories con `logits[:, 63:-1, :]` hardcodeado (frágil si cambia `seq_len`).
- Seed única (42) en todos los experimentos: sin barras de error ni réplicas.
- Un checkpoint binario commiteado al repo (`e884aa4`) + ~22 checkpoints sueltos sin `.gitignore` (riesgo de bloat del repo).
- `spec_wave/model.py:61`: la medición de latencia incluye `argmax` dentro del cronómetro (trivial, pero sesga).

---

## 5. Evaluación de los Experimentos

### 5.1 Sólido y valioso

| Elemento | Por qué importa |
| :--- | :--- |
| Control del suelo entrópico (§2) | Misma data, arquitecturas de 21M a 123M, mismo suelo → evidencia fuerte de límite informacional. Con 3 seeds + CIs: **negative result publicable en workshop** (p. ej. *"One-shot parallel decoding hits a capacity-independent entropy floor"*). |
| Pase de honestidad post-19/08 | Reinterpretación correcta de la "phase transition" (memorización), speedups extrapolados marcados como tales. Estándar científico ejemplar. |
| Pipeline de datos | Streaming no-repetitivo, splits ciegos, baselines medidos en los mismos splits. |

### 5.2 No sólido

- **Todo lo de latencia/speedup** salvo la medición 0.76x (negativa y válida). Los 104x–155x derivan de constantes arbitrarias; los baselines carecen de KV-cache.
- **El veredicto "la difusión falla"** proviene de un denoiser MLP minúsculo con T=50, sin atención profunda ni schedules modernos. Válido como *gate* de falsación rápida; prematuro como conclusión sobre la vía difusional. La literatura actual (MDLM, SEDD, LLaDA, Mercury/Gemini-Diffusion) hace difusión **discreta sobre tokens**, no continua sobre espectros de 8,192 dims — la Fase 3 probó la variante más difícil.

---

## 6. Vías de Rescate (beneficio potencial real)

Priorizadas por (probabilidad de resultado positivo) × (interés):

### 6.1 Curva semi-AR M∈{1,2,4,8,16} + refinamiento intra-ráfaga tipo mask-predict ⭐

Dentro de cada ráfaga: emitir → enmascarar la mitad con menor confianza → re-emitir condicionando en lo visible, 2–4 rondas (estilo CMLM, Ghazvininejad et al.). Restaura dependencia condicional manteniendo paralelismo parcial. `WaveletBurstDecoder` lo soporta con cambios menores.

**Entregable:** gráfico PPL vs nº de forwards que caracteriza el trade-off entropía/velocidad. Contribución empírica legítima con baseline claro (blockwise parallel decoding, Stern et al. 2018).

### 6.2 Speculative decoding bien hecho (drafter MTP barato) ⭐

La cabeza de ráfaga es esencialmente un *multi-token prediction head* (familia Medusa / EAGLE / DeepSeek-MTP / Meta-MTP). Requisitos:

1. Drafter sobre backbone pequeño/congelado o cabeza ligera sobre hidden states cacheados (≥8x más barato que el verificador).
2. KV-cache en draft y verify.
3. α contado correctamente (histograma de posición de primer rechazo).
4. Objetivo: α ≥ 60% con K=4 → 1.5–2.2x real.

**Punto clave:** el drafter **no necesita** PPL cercano al verificador — necesita ser *barato y decente*. Este pivote convierte el "fracaso del generador" en "drafter útil".

### 6.3 MTP como pérdida auxiliar para representaciones

Probar si el objetivo espectral/ráfaga como *aux loss* mejora un LM causal pequeño en downstream (hipótesis de multi-token prediction, LeCun et al. 2024). Experimento barato, con baseline claro, publicable sea cual sea el resultado.

### 6.4 Coarse-to-fine en espacio wavelet

Generar LL primero y refinar detalle condicionado en pasadas posteriores. La única versión donde la estructura wavelet aporta algo real (jerarquía explícita de escala).

### 6.5 Negative result formal

El hallazgo del §2 como paper de workshop con seeds múltiples y CIs.

### 6.6 Lo que se cerraría

- El one-shot O(1) como generador general (evidencia + teoría lo sellan).
- Cualquier claim de speedup sin baseline KV-cache medido.

---

## 7. Plan de Acción Sugerido (1 semana)

1. **Ablación CE-only vs híbrida** en `NativeSpecWaveLM` (horas, CPU). Podría mover el PPL más que cualquier otra cosa pendiente.
2. **Corregir métrica α** + histograma de rechazos en el benchmark especulativo.
3. **Sweep semi-AR** M∈{2,4,8} × rondas mask-predict ∈{1,2,4}, mismo split TinyStories, **3 seeds**, reportando PPL y nº de forwards.
4. **Benchmark especulativo v2**: `distilgpt2` como backbone del drafter + KV-cache en ambos lados, vs GPT-2 KV-cacheado.
5. **Higiene del repo**: `.gitignore` para `checkpoints/*.pt`, `data/`, `__pycache__/`; mover el checkpoint commiteado a releases/HF Hub.

---

## 8. Reflexión Final

La intuición de que "la línea puede no ser viable en su objetivo principal pero requiere investigarse por beneficios potenciales" es exactamente correcta, y el repositorio ya contiene la prueba: el proyecto pasó de intentar demostrar 250x a **falsar su propia hipótesis con rigor** — y esa falsación produjo el hallazgo más interesante (el suelo entrópico independiente de capacidad) y dos pivotes concretos (drafter MTP barato, semi-AR con refinamiento iterativo) alineados con donde está la industria ahora mismo.

La idea como "generador one-shot" está muerta. Como "cabeza paralela multi-token barata para acelerar modelos causales", tiene una vía clara, medible y con precedentes industriales.
