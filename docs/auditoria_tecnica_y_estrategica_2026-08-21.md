# Auditoría técnica y estratégica de SpecWave

**Fecha:** 2026-08-21  
**Alcance:** hipótesis de investigación, arquitectura, datos, entrenamiento, evaluación, benchmarks y artefactos disponibles en el repositorio.

> ⚠️ **Nota:** existe una síntesis conjunta de esta auditoría con la de ox-alpha/opencode en [`docs/auditoria_conjunta_2026-08-21.md`](auditoria_conjunta_2026-08-21.md) — usar ese documento como referencia accionable principal.

> **Veredicto breve:** la línea merece investigación, pero la hipótesis de que la transformada wavelet exacta permita reemplazar un modelo autoregresivo de calidad comparable por una emisión lingüística de un paso no está respaldada. La evidencia local más reciente apunta a un límite estructural de la generación no autoregresiva de bloques, no a una mera falta de entrenamiento. La ruta con mayor potencial práctico es la decodificación por bloques/refinamiento iterativo o el *speculative decoding*, no el objetivo de un LLM general O(1).

---

## 1. Qué se ha verificado

### 1.1 Transformada Haar 2D

`spec_wave/wavelet.py` implementa una DWT/IDWT de Haar 2D correcta para tensores de tamaño par. La prueba de reconstrucción y conservación de energía pasa a precisión numérica: el error de reconstrucción observado es del orden de `4.77e-07`.

Esto verifica una propiedad algebraica útil: la representación es una biyección isométrica. **No verifica** que las subbandas tengan semántica lingüística, capacidad generativa adicional ni compresión.

### 1.2 Emisión paralela de un bloque fijo

Las arquitecturas del repositorio sí emiten todos los logits de una secuencia de longitud fija mediante un número constante de iteraciones de decodificación. Es más preciso llamarlo *decodificación de profundidad constante para longitud de bloque fija*.

No debe llamarse coste O(1) sin matices:

- el cálculo, la memoria y la proyección al vocabulario escalan con la longitud del bloque;
- para una respuesta arbitrariamente larga se necesitan varios bloques;
- un bloque de logits de tamaño `L × |V|` no tiene coste constante;
- el código ejecuta múltiples operadores, no un único kernel GPU.

### 1.3 Datos y generalización

Los experimentos históricos mezclan tres niveles de evidencia:

1. **Sintético o hardcodeado:** recuperación/memorización y funciones deterministas. Útiles para depurar mecánica, no para demostrar lenguaje.
2. **Autoencoder:** reconstruir tokens al recibir esos mismos tokens o sus embeddings. Mide capacidad de reconstrucción, no generación condicionada.
3. **TinyStories train/validation:** es la evidencia más útil disponible para el modelo nativo y para el decodificador por ráfagas. El train y validation proceden de splits distintos del dataset.

El resultado WikiText-2 de adaptación sobre pocos pares también es informativo por ser negativo: PPL de train cercana a 1 y PPL de validación de miles es memorización severa, no una transición de fase ni evidencia de razonamiento espectral.

---

## 2. Diagnóstico de la arquitectura nativa

El modelo actual `NativeSpecWaveLM` es, operacionalmente, un encoder–decoder no autoregresivo condicionado en el prompt:

1. embeddings y posición del prompt;
2. DWT 2D;
3. concatenación y proyección de subbandas;
4. Transformer encoder sobre el prompt;
5. Transformer decoder desde queries de salida aprendidas;
6. cuatro cabezas de subbandas, IDWT, refiner y cabeza de vocabulario atada.

La DWT no crea un cuello de botella. Para `d_model = d` y 64 posiciones:

```text
entrada espacial:       64 × d
subbandas concatenadas: 32 × (4 × d/2) = 64 × d
salida de subbandas:    4 × 32 × d/2 = 64 × d
IDWT:                   64 × d
```

La DWT/IDWT es por tanto un cambio de base ortogonal y exacto. Si una red posterior tiene capacidad suficiente, puede aprender una transformación equivalente en base plana. La evidencia de la ablation histórica, donde la variante plana y la wavelet son aproximadamente equivalentes, es coherente con esta observación.

### Consecuencia

Hoy no hay evidencia de que `LL`, `LH`, `HL` y `HH` correspondan a semántica, sintaxis o detalle léxico. Esas etiquetas deben tratarse como hipótesis, no como propiedades del modelo. Para sostenerlas haría falta una intervención causal: suprimir, intercambiar o cuantizar cada subbanda y medir efectos semánticos y sintácticos en un test controlado.

---

## 3. El límite de PPL es principalmente estructural

Un modelo autoregresivo factoriza:

\[
p(y_{1:L}\mid x) = \prod_i p(y_i\mid x, y_{<i}).
\]

El modelo one-shot de SpecWave aproxima una factorización condicionalmente independiente:

\[
p(y_{1:L}\mid x) \approx \prod_i p(y_i\mid x).
\]

En lenguaje, el prompt no determina de forma única 64 tokens futuros. Hay muchas continuaciones plausibles, pero sus tokens están fuertemente correlacionados. Predecir las posiciones en paralelo elimina la información que los primeros tokens aportan a los siguientes. El resultado esperado es una combinación de tokens frecuentes o localmente plausibles sin una continuación global concreta.

Esto explica mejor el patrón observado que una deficiencia específica de Haar:

- cierta exactitud de tokens frecuentes/estructura superficial;
- PPL muy superior a GPT-2 causal;
- estancamiento al escalar pasos de optimización;
- mejora parcial cuando se introducen ráfagas semi-autoregresivas.

El PPL de un modelo NAR sigue siendo una métrica propia válida, pero no es una comparación directa de la misma factorización que el PPL autoregresivo. Cualquier comparación debe incluir baselines NAR/iterativos de capacidad y presupuesto equivalentes.

---

## 4. Evidencia de entrenamiento reciente

Los checkpoints locales `native_specwave_decay_200m*` contienen un modelo de **53,419,776 parámetros**. Los metadatos disponibles muestran:

| Paso | Tokens procesados declarados | PPL validación ponderada |
|---:|---:|---:|
| 4,000 | 16,384,000 | 374.66 |
| 10,000 | 40,960,000 | 376.27 |
| 16,000 | 65,536,000 | 370.36 |
| 28,000 | 114,688,000 | 370.13 |
| 36,000 (mejor) | 147,456,000 | 369.51 |
| 40,000 | 163,840,000 | 369.73 |

La mejora tras los primeros millones de tokens es marginal y no monótona. Este resultado no sugiere que ejecutar más épocas sobre el mismo subconjunto sea la palanca principal.

### Dos matices necesarios

1. `tokens_seen = steps × batch_size × 128` cuenta tokens procesados. No demuestra que sean tokens **únicos**: el dataset se materializa una vez y las épocas posteriores repiten sus mismos pares.
2. La variante de decaimiento temporal informa una pérdida ponderada hacia posiciones tempranas. Su PPL no equivale al PPL estándar de CE uniforme ni debe compararse directamente con el valor de GPT-2. Es una métrica diagnóstica de horizonte, no una mejora global demostrada.

---

## 5. Auditoría de benchmarks y métricas

### 5.1 Tests oficiales

`tests/test_core.py` pasa actualmente, pero sólo confirma:

- inversión numérica DWT/IDWT;
- ejecución de un forward de un modelo pequeño;
- aprendizaje de la función sintética `(prompt * 3 + 7) % vocab`.

El test de latencia medido durante la auditoría fue **11.97 ms**, mientras que el mensaje del propio test afirma “sub-millisecond”. No hay baseline ni aserción de latencia. Debe renombrarse como *smoke benchmark* y eliminar esa afirmación.

### 5.2 Benchmarks de velocidad

`tests/benchmark_gpu_wallclock.py` mide algunos puntos, pero para longitudes 128 y 256 sustituye el baseline por la constante arbitraria `13.0 ms/token`. Los speedups derivados de esa extrapolación no son resultados experimentales.

Para una comparación válida se necesita:

- mismo hardware y precisión;
- sincronización CUDA antes/después de temporizar;
- warmup y percentiles p50/p95;
- GPT-2 u otro baseline real con KV-cache;
- misma longitud de prompt, batch, muestreo y longitud de salida;
- calidad comparable o una curva calidad/latencia.

### 5.3 Checkpoint y speculative decoding

El checkpoint `specwave_tinystories_burst.pt` fue entrenado con ráfagas de 16 tokens. El benchmark especulativo usa por defecto ráfagas de 8 y hace carga parcial por coincidencia de forma. Así se descartan parámetros de forma incompatible, incluido el query positional de la ráfaga. Por tanto, el benchmark no evalúa exactamente el drafter entrenado.

Además, su contador de “acceptance rate” suma tokens emitidos tras una sustitución por rechazo y/o un bonus token. Debe registrar por separado:

- propuestas de draft;
- propuestas aceptadas literalmente;
- tokens de reemplazo;
- bonus tokens;
- tokens finales por paso de verificación.

Un drafter útil ha de evaluarse también en coste: la implementación actual realiza un forward del backbone para proponer y otro del modelo objetivo para verificar. Sin KV-cache y sin alta aceptación, es probable que pierda frente a la generación causal normal.

### 5.4 Documentación

El README y algunos informes ya contienen advertencias honestas. Sin embargo, `docs/spec_wave_paper_draft.md` conserva claims que contradicen los scripts y las notas críticas: “lossless recovery” general, speedups extrapolados y seguridad semántica. No debe circular como borrador científico hasta sustituir esos resultados por mediciones reproducibles o etiquetarlos como obsoletos.

---

## 6. Decisiones de investigación recomendadas

### No priorizar ahora

- otro entrenamiento largo del mismo modelo y del mismo subconjunto;
- disminuir PPL mediante ponderación de horizonte y reportarlo como PPL estándar;
- claims de speedup frente a LLMs reales sin KV-cache y calidad equivalente;
- atribuir semántica a subbandas por su nombre.

### Experimentos prioritarios

1. **Ablation nativa decisiva.** Misma arquitectura, número de parámetros, datos, tokens de cómputo y tres semillas; sustituir DWT/IDWT por reshape/proyecciones planas. Reportar CE estándar, PPL, exactitud por posición, coste y media ± desviación.

2. **Ablation del objetivo.** Comparar CE sola, CE + loss espectral, CE + MSE de embedding, y embeddings/cabeza atados o no atados. Las pérdidas sobre embeddings aprendidos pueden restringir la optimización sin mejorar la distribución discreta.

3. **Escalar datos de verdad antes que épocas.** Guardar el identificador/revisión del dataset, el número de ejemplos únicos y el hash de la lista de pares. Usar un flujo que aporte ejemplos nuevos por época o un corpus materializado mayor.

4. **Mover el objetivo a generación iterativa.** Evaluar ráfagas de 2, 4, 8 y 16 tokens, con *scheduled sampling* para reducir exposición al error. Comparar contra un baseline causal y un baseline NAR sin wavelets.

5. **Ruta speculative decoding.** Entrenar el drafter a aproximar la distribución del verificador (KL sobre logits, no sólo CE contra el token observado); usar el mismo `burst_len` al entrenar y servir; medir aceptación literal y latencia real con KV-cache.

6. **Sólo entonces evaluar wavelets como señal científica.** Aplicar intervenciones de subbandas y medir si existe una ventaja repetible sobre el baseline plano. Si no aparece, conservar Haar como implementación opcional, no como tesis central.

---

## 7. Formulación viable de la contribución

La contribución defendible no es aún “un LLM que sustituye la autoregresión en un paso”. Las formulaciones más realistas son:

- un estudio negativo y reproducible sobre el límite de la emisión one-shot de bloques de texto;
- un decodificador de ráfagas condicionado para reducir pasos secuenciales;
- un drafter multiescala para speculative decoding, si mejora aceptación y wall-clock;
- una representación wavelet útil sólo si demuestra un beneficio mediante ablaciones estrictas;
- un autoencoder/compresor de embeddings, claramente separado de generación de lenguaje.

## 8. Criterio de continuación

La línea sigue siendo viable si, tras la ablation nativa y el baseline NAR, aparece al menos una de estas señales:

- ventaja consistente de calidad/coste de las wavelets frente a la base plana;
- mejora de PPL estándar al aumentar iteraciones de refinamiento a igual presupuesto;
- acceptance rate y throughput reales superiores en speculative decoding;
- evidencia causal de que las subbandas soportan una operación útil (coarse-to-fine, compresión, control o denoising).

Si no aparece ninguna, el resultado sigue siendo valioso: falsaría de forma limpia la hipótesis de que una transformación wavelet exacta por sí misma supera la dependencia autoregresiva en lenguaje.
