# 🔍 SpecWave: Auditoría Conjunta y Backlog Accionable

> **STATUS: [SÍNTESIS DE DOS AUDITORÍAS INDEPENDIENTES / 2026-08-21 / DOCUMENTO PRINCIPAL]**
> Este documento fusiona y reconcilia dos auditorías independientes del proyecto:
> - **Auditoría A:** `docs/spec_wave_audit.md` (ox-alpha/opencode) — análisis cuantitativo del límite de PPL, auditoría de código con referencias línea a línea.
> - **Auditoría B:** `docs/auditoria_tecnica_y_estrategica_2026-08-21.md` (GPT 5.6 Earth) — crítica conceptual, inspección de checkpoints en curso, higiene métrica y criterios de continuación.
>
> Ambas convergen en todas las conclusiones mayores. Los hallazgos exclusivos de cada una están marcados con **[A]**, **[B]** o **[A+B]**. Los bugs marcados ✅ han sido verificados en el código.

---

## 1. Veredicto Ejecutivo [A+B]

| Dimensión | Evaluación |
| :--- | :--- |
| **Hipótesis central** (generador NAR one-shot competitivo en PPL) | ❌ **Cerrada por evidencia propia + teoría.** Suelo entrópico informacional, confirmado por 5 arquitecturas (A) y por el experimento en curso en vivo (B). |
| **"O(1)" como claim** | ⚠️ Es *decodificación de profundidad constante para bloque de longitud fija*: cómputo, memoria y proyección a vocabulario escalan con L, y respuestas largas requieren múltiples bloques (B). |
| **Neutralidad de las wavelets** | ✅ Reparametrización ortogonal exacta; la ablación B≈A era matemáticamente inevitable (A+B). La semántica LL/LH/HL/HH es hipótesis sin evidencia causal (B propone el protocolo). |
| **Calidad metodológica** | ✅ Alta tras el pase de honestidad del 19/08: splits ciegos, datos streaming no-solapados, baselines medidos. |
| **Hallazgo más valioso** | El **suelo entrópico independiente de capacidad** — publicable como negative result con controles (seeds, CIs). |
| **Vías de rescate** | 3 pivotes medibles alineados con la industria: drafter MTP barato (speculative decoding), decodificación iterativa intra-bloque, coarse-to-fine wavelet. |

---

## 2. Por Qué el PPL No Baja: el Suelo Entrópico [A+B]

### 2.1 Evidencia histórica entre arquitecturas [A]

Cinco arquitecturas radicalmente distintas convergen a la misma meseta de validación:

| Arquitectura | Params | Datos (val blind) | Val PPL | Fuente |
| :--- | ---: | :--- | ---: | :--- |
| MLP plano (sin wavelets) | 21.7M | WikiText 512k tok | 1,088.57 | `findings_phase2_ablation.md` |
| Transformer cruzado profundo | 109.5M | WikiText 1.5k pares | 1,108.30 | `findings_deep_gpt2_spectral_transformer.md` |
| GPT-2 (6-11) + espectral | 123.6M | WikiText 2.3M tok | 535.33 | `findings_massive_scale_training.md` |
| Nativo from-scratch | 34.1M | TinyStories 7.68M tok | 337.45 | `findings_native_specwave_from_scratch.md` |
| Difusor DDIM 10 pasos | 23.4M | WikiText 512k tok | Acc 1.30% | `findings_phase3_diffusion_refinement.md` |

Train ≈ Val en todos los experimentos a escala: el límite es **informacional**, no de capacidad ni optimización. Multiplicar params ×5 movió el PPL un 2%.

### 2.2 Evidencia en vivo (experimento decay 200M) [B]

Inspección de los checkpoints locales `native_specwave_decay_200m*` (53.4M params):

| Paso | Tokens procesados | PPL val (ponderada) |
|---:|---:|---:|
| 4,000 | 16.4M | 374.66 |
| 10,000 | 41.0M | 376.27 |
| 16,000 | 65.5M | 370.36 |
| 28,000 | 114.7M | 370.13 |
| 36,000 (mejor) | 147.5M | 369.51 |
| 40,000 | 163.8M | 369.73 |

Plano y no monótono desde los primeros millones de tokens: más épocas sobre el mismo subconjunto no es la palanca. Dos matices de higiene [B]:
1. `tokens_seen = steps × batch × 128` cuenta tokens **procesados**, no únicos; con épocas sobre un dataset materializado una vez, hay repetición masiva.
2. La PPL ponderada por horizonte ($\gamma=0.94$) **no es comparable** ni con CE uniforme ni con GPT-2: es métrica diagnóstica de horizonte, no PPL estándar.

### 2.3 Teoría [A+B]

La decodificación one-shot factoriza $P(y_{1:L}|X) \approx \prod_i P(y_i|X)$; el óptimo de Bayes de esa familia es la suma de entropías marginales (~7.0 nats WikiText, ~5.8 TinyStories). Las mesetas observadas **son** ese óptimo. Consecuencias:

- Intentar bajar el PPL one-shot con ingeniería es matemáticamente estéril.
- La reponderación de horizonte solo redistribuye error (frente ↑, cola ↓), no crea información condicional.
- El PPL NAR no es directamente comparable al AR: refleja la restricción de factorización, no solo calidad del modelo [B]. Comparaciones válidas requieren baselines NAR/iterativos de igual presupuesto.
- Única salida: **cambiar la factorización** (refinamiento iterativo, ráfagas pequeñas, mask-predict).

---

## 3. Wavelets: Neutrales por Construcción [A+B]

1. $\text{IDWT} \circ \text{MLP} \circ \text{DWT}$ con Haar fija es reparametrización del MLP; B≈A ($\Delta$ Loss 0.0001) era inevitable.
2. El "vocoder" de la Fase 1 contiene DWT→IDWT = identidad: la Puerta 1 se pasa por construcción.
3. Sin compresión real: entrada y espectro son ambos 64×d (análisis dimensional de B).
4. El emparejamiento 2D (tokens×canales) es arbitrario; permutar filas/columnas da otra descomposición equally válida.

**Protocolo para dejar de ser hipótesis [B]:** intervención causal sobre subbandas — suprimir, intercambiar o cuantizar cada una y medir efectos semánticos/sintácticos en test controlado. Solo si existe ventaja repetible sobre base plana, Haar pasa de implementación opcional a tesis central.

**Único diseño donde aportan estructura real [A]:** generación coarse-to-fine sobre escalas (LL primero, refinar detalle condicionado después) — equivalente a difusión en cascada / generación progresiva.

---

## 4. Auditoría de Código Consolidada

### 4.1 Fortalezas [A]
DWT/IDWT correctos (Parseval a precisión de máquina), datasets no-solapados con splits ciegos, baselines nativos medidos en los mismos splits, scripts reproducibles.

### 4.2 Hallazgos (ordenados por gravedad)

**A1. El drafter especulativo no es barato — fallo de diseño** [A+B] ✅
`examples/benchmark_wavelet_speculative_decoding.py:46-65`: envuelve **el mismo backbone GPT-2 que el verificador**. Cada paso cuesta 2 forwards completos → speedup wall-clock máximo <2x incluso con aceptación perfecta. El 0.76x medido es coherente.

**A2. Mismatch de `burst_len`: el benchmark no evaluó el drafter entrenado** [B] ✅ verificada por A
El checkpoint `specwave_tinystories_burst.pt` se entrenó con `burst_len=16` (`train_scaled_tinystories_burst.py:79`) pero el benchmark usa K=8 por defecto y carga pesos filtrando por forma → `query_pos` se descarta silenciosamente. El α=18.39% y el 0.76x se midieron sobre un drafter parcialmente inicializado. **Re-ejecutar desde cero.**

**A3. Bug en la tasa de aceptación** [A+B] ✅
`benchmark_wavelet_speculative_decoding.py:120-131`: el token de reemplazo cuenta como aceptado (`num_accepted += 1`) y el bonus entra en el numerador. El "12.50%" del drafter sin entrenar es exactamente 1/8 (rechazo inmediato en cada paso). Registrar por separado: propuestas, aceptaciones literales, reemplazos, bonus, tokens finales por paso [B].

**A4. Ningún lado usa KV-cache** [A+B] ✅
Baseline AR reintroduce contexto completo cada paso; verify también. Todo speedup compara contra un baseline desventajado. Protocolo mínimo exigible [B]: mismo hardware/precisión, sync CUDA, warmup, percentiles p50/p95, baseline real con KV-cache, misma longitud/batch/muestreo, curva calidad-latencia.

**A5. Pérdida híbrida posiblemente contraproducente — ablación clave ausente** [A+B]
`spec_wave/native_model.py:136-152`: `CE + 2·spectral_MSE + 2·manifold_MSE` arrastra `refined` hacia `wte[target]`; con weight-tying (línea 88) entrena medio autoencoder en vez de likelihood. Comparar: CE sola / CE+espectral / CE+MSE embedding / atado vs no atado.

**A6. Coste oculto y ganancia predecible del semi-AR** [A]
`train_semiautoregressive_specwave.py:149-163`: cada ráfaga re-codifica contexto completo sin caché ("16x" necesita asterisco). PPL 334 vs 337 one-shot es teóricamente inevitable: solo 4/64 posiciones reciben condicionamiento fresco.

**A7. Test de latencia con claim sin respaldo** [B]
`tests/test_core.py` imprime "sub-millisecond" pero midió 11.97 ms durante la auditoría; no hay aserción de latencia. Renombrar a *smoke benchmark* y eliminar el claim.

**A8. Menores**
- Paper draft conserva claims contradictorios con los scripts; etiquetar obsoleto o sustituir antes de circular [B].
- Seed única (42) en todos los experimentos: sin barras de error [A].
- Baseline TinyStories con `logits[:, 63:-1, :]` hardcodeado [A].
- PPL capado a exp(20) en evaluadores (documentar) [A].
- Higiene de repo resuelta el 2026-08-21: `.gitignore` creado, checkpoints/data des-trackeados, blob de 694 MB purgado de la historia local [A].

---

## 5. Evaluación de Experimentos [A+B]

**Sólido:** control del suelo entrópico (5 arquitecturas, mismo split); pase de honestidad post-19/08; pipeline de datos streaming con splits ciegos.

**No sólido:** latencia/speedup salvo el 0.76x (negativo y válido); el veredicto "la difusión falla" proviene de un denoiser minúsculo T=50 sin atención profunda — válido como gate de falsación, prematuro como conclusión (la vía moderna es difusión **discreta** sobre tokens: MDLM/SEDD/LLaDA, no continua sobre espectros de 8,192 dims).

---

## 6. Vías de Rescate Priorizadas

### R1. Decodificación iterativa intra-bloque (semi-AR mejorada) [A+B]
Curva M∈{1,2,4,8,16} con **mask-predict intra-ráfaga** (enmascarar posiciones de baja confianza y re-emitir condicionando en lo visible, 2–4 rondas; estilo CMLM) [A] ± *scheduled sampling* contra exposure bias [B]. Entregable: gráfico PPL estándar vs nº de forwards. Baselines: causal y NAR plano a igual presupuesto.

### R2. Speculative decoding bien hecho [A+B]
La cabeza de ráfaga es un *multi-token prediction head* (familia Medusa/EAGLE/DeepSeek-MTP). Requisitos consolidados:
1. Drafter ≥5–10x más barato que el verificador (`distilgpt2` o cabeza ligera sobre hidden states cacheados).
2. KV-cache en draft y verify.
3. Mismo `burst_len` en entrenamiento y servicio (lección A2).
4. Entrenar el drafter con **KL sobre logits del verificador**, no solo CE contra token observado [B].
5. Métricas corregidas (A3): α literal, histograma de posición de primer rechazo, tokens/step, wall-clock real.
Objetivo: α≥60% con K=4 → 1.5–2.2x real. El drafter no necesita PPL cercano al verificador: necesita ser barato y decente.

### R3. Ablaciones decisivas (baratas, hacer primero) [A+B]
(a) CE-only vs híbrida; (b) wavelet vs reshape/plano en la arquitectura nativa, mismas params/datos/tokens, **3 seeds**, reportando media±desviación, CE estándar, exactitud por posición.

### R4. MTP como pérdida auxiliar para representaciones [A]
Probar si el objetivo espectral/ráfaga como aux loss mejora un LM causal pequeño downstream (hipótesis multi-token prediction). Barato y publicable sea cual sea el resultado.

### R5. Coarse-to-fine en espacio wavelet [A]
LL-first con refinamiento de detalle condicionado; única versión donde la jerarquía wavelet aporta estructura explícita. Validar primero con intervenciones de subbandas [B].

### R6. Negative result formal [A+B]
"Suelo entrópico independiente de capacidad en decodificación one-shot" — workshop-grade con seeds múltiples y CIs.

### No priorizar ahora [B]
Otro entrenamiento largo del mismo modelo/subconjunto; reportar PPL ponderada como estándar; claims de speedup sin KV-cache; atribuir semántica a subbandas por su nombre.

---

## 7. Criterios de Continuación (falsables) [B]

La línea sigue viva si aparece alguna de estas señales:
1. Ventaja consistente calidad/coste de wavelets vs base plana (R3/R5).
2. Mejora de PPL estándar al aumentar iteraciones de refinamiento a igual presupuesto (R1).
3. α y throughput reales superiores en speculative decoding (R2).
4. Evidencia causal de operación útil sobre subbandas (compresión, control, denoising).

Si no aparece ninguna: el resultado limpio que falsa la hipótesis también es valioso.

---

## 8. Backlog Accionable Priorizado

| # | Acción | Coste | Dependencia |
|---|:---|:---|:---|
| 1 | Ablation CE-only vs híbrida (R3a) | Horas, CPU | Ninguna |
| 2 | Fix `burst_len` (A2) + métrica α (A3) + histograma de rechazos | Horas | Ninguna |
| 3 | Re-benchmark especulativo v2: drafter barato + KL + KV-cache (R2) | 1-2 días GPU/CPU | #2 |
| 4 | Sweep semi-AR M×rondas mask-predict, 3 seeds (R1) | Días CPU | Fin del run actual |
| 5 | Ablation nativa wavelet vs plano, 3 seeds (R3b) | Días CPU | Fin del run actual |
| 6 | Renombrar smoke benchmark + etiquetar paper draft obsoleto (A7/A8) | Minutos | Ninguna |
| 7 | Intervenciones causales de subbandas (R5 validación) | Días | Tras #5 |

---

## 9. Reflexión Final [A+B]

Ambas auditorías, producidas de forma independiente, señalan lo mismo: la falsación rigurosa del objetivo original produjo el hallazgo más interesante del proyecto (el suelo entrópico) y dos pivotes concretos alineados con la práctica industrial actual. La idea como "generador one-shot" está cerrada; como "cabeza paralela multi-token barata para acelerar modelos causales" tiene una vía clara, medible y con precedentes.
