# SpecWave: Plan de Investigacion Experimental

## Objetivo

Determinar si la generacion de lenguaje no-autorregresiva (NAR) con wavelets 2D y un vocoder paralelo es viable.

Este plan organiza la idea en experimentos con hipotesis falsables, metricas y puertas de decision (go/no-go). El objetivo es fallar pronto, no demostrar que la idea funciona.

---

## Marco: que es realmente SpecWave

1. No hay compresion. El espectro de salida tiene la misma dimension que la entrada (N x d). El razonador MLP aprende un mapa N x d -> N x d. Esto escala mal. Es el riesgo principal.

2. El "O(1)" no es gratis. Un forward con salida N x d acoplado a un vocabulario de 50k simbolos crece con N. La generacion es de longitud fija, sin parada dinamica.

3. La analogia con el audio es parcial. En audio, el mel-spectrogram es una representacion comprimida. Aqui, el espectro wavelet es solo una reparametrizacion lineal de los embeddings. La via real es la generacion NAR iterativa (diffusion, masked LM), no un "disparo unico".

**Hipotesis central:** La parte mecanica (el vocoder) ya funciona. Hay que probar si un generador puede producir un espectro coherente y no memorizado para texto nuevo.

---

## Fase 0: Fundamentos (VERIFICADO)

- La transformada de Haar 2D es una biyeccion isometrica exacta (error Parseval ~6e-08).
- El vocoder puede memorizar corpora pequenos (100% exact match).
- Un forward unico es barato (7-28 ms para N=32-256).

Conclusion: la mecanica wavelet funciona. El problema esta en la generalizacion del generador.

---

## Fase 1: Invertibilidad a escala (VERIFICADO / PUERTA 1 PASADA)

**Hipotesis:** El vocoder puede reconstruir (no memorizar) bloques de texto nunca vistos, con datos reales.

**Experimento:**
1. Dataset: WikiText-2 real de HuggingFace/PyTorch (patron en `benchmarks/phase1_vocoder_scale.py`).
   - Train: 10,000 bloques de 64 tokens (640k tokens).
   - Test: 1,000 bloques blind (64k tokens).
2. Protocolo: tokens -> embeddings -> DWT 2D -> IDWT -> logits. El target es el mismo bloque.
3. Metricas: Exact token match 97.45%, Exact sequence match 23.50%, PPL de reconstruccion test **1.4568** (Loss 0.3762).

**Puerta 1:** ✅ **PASADA** (PPL test 1.4568 $\le 2.0$ y exactitud de token 97.45% $\ge 95\%$). Proceder a Fase 2.

---

## Fase 2: Ablacion (VERIFICADO / GATE 2: B ≈ A)

**Hipotesis:** Las wavelets aportan una ventaja real al generador.

**Experimento:**
- A: Generador con wavelets 2D (SpecWave: 21.72M params).
- B: Generador sobre la secuencia plana $N \times D$ sin wavelets (Flat Baseline: 21.72M params).
- Dataset: WikiText-2 real (4,000 pares train, 500 pares blind test).

**Resultados Medidos:**
- Modelo A (SpecWave): Val Loss **6.9927** | Val PPL **1088.70** | Val Acc **5.03%**
- Modelo B (Flat): Val Loss **6.9926** | Val PPL **1088.57** | Val Acc **5.03%**
- Diferencia: $\Delta \text{Loss} = -0.0001$, $\Delta \text{PPL} = -0.13$.

**Criterio / Veredicto:** ⚠️ **NEUTRAL ($B \approx A$)**: La transformada de Haar 2D es una rotación ortogonal exacta en el espacio de Hilbert; las capas densas aprenden un subespacio isomórfico con y sin descomposición wavelet. El cuello de botella de la generación NAR no es la base wavelet, sino el determinismo de 1 solo disparo. Proceder a Fase 3 (Refinamiento Iterativo / Difusión).

---

## Fase 3: Generacion condicionada & Difusión (VERIFICADO / GATE 3 EVALUADO)

**Hipotesis:** Un generador o difusor en el espacio espectral wavelet 2D produce una continuación de texto coherente para un prompt dado.

**Experimentos y Resultados:**
1. **Generador 1 paso (MLP / Feedforward):** Val PPL $\approx 1088$, Val Token Acc $5.03\%$ (colapso multimodal del disparo único).
2. **Difusor Espectral Continuo (Diffusion-LM en 2D DWT, 10-step DDIM):** Val Gen Acc $1.30\%$, Noise MSE $1.0006$.

**Conclusión:** Generar bloques completos no-autorregresivos en un espacio continuo de $8,192$ dimensiones sin atención autorregresiva profunda sufre de entropía multimodal severa. La mecánica del vocoder (Fase 1) es excelente como autoencoder ($97.45\%$ exactitud), pero la generación libre no generaliza a nivel de lenguaje natural.

---

## Resumen de experimentos y Estado Real

| # | Experimento | Pregunta clave | Puerta para avanzar | Estado Real Medido |
|---|------------|---------------|---------------------|--------------------|
| 1 | Invertibilidad real (WikiText-2) | ¿El vocoder supera la memorizacion? | PPL test < 2.0 | ✅ **PASADO** (PPL 1.46, Acc 97.45%) |
| 2 | Ablacion wavelets vs MLP plano | ¿Aportan algo real? | A > B en PPL de test | ⚠️ **VERIFICADO** (B ≈ A, equivalencia ortogonal) |
| 3 | Generacion condicionada & Difusión | ¿Funciona la via NAR iterativa? | PPL test < 50 / mejora sobre 1 paso | ⚠️ **EVALUADO** (Acc 1.30%, reto dimensional de bloque) |
| 4 | Latencia GPT-2 real (KV-cache) | ¿Es rapida a igual calidad? | Speedup real > 2x | ⚠️ Condicionado a calidad de generación |
| 5 | Coherencia e interpretacion | ¿Genera parrafos coherentes? | Preferencia humana > 50% | ❌ No aplicable por límite de generalización |

## Orden de ejecucion

1 -> 2 -> 3 (primero el 3.3) -> 4 -> 5 -> 6.

Los primeros experimentos son baratos (horas). El 3.3 es el que decide todo.

---

## Expectativa realista

1. El disparo unico con MLP plano probablemente no funcione (no hay historial de exito en la literatura).
2. La via NAR iterativa (diffusion) es la mas prometedora. SpecWave quedaria como "diffusion sobre espectros wavelet". El interes real seria la representacion intermedia.
3. La subbanda LL como "semantica" no esta probada con texto real: la energia de Haar depende del embedding, no del significado.
4. Si las fases 2-3 fallan, el vocoder seria mas util como compresor / autoencoder de texto, no como generador.
5. Presupuesto: 5-6 experimentos, una persona, una GPU (T4/RTX), de horas a unas pocas semanas.

---

## Cambios vs el roadmap original

| Aspecto | Roadmap original (romantico) | Este roadmap (critico) |
|---------|------------------------------|--------------------------|
| Objetivo | Probar 250x, lossless, etc | Fallar la idea pronto |
| Fase 1 | FineWeb 100k | WikiText 50k/5k |
| Fase 2 | Pre-training TinyStories | Ablacion wavelets vs MLP |
| Fase 3 | Wall-clock 250x | El problema real: generacion |
| Fase 4 | Safety | Solo si hay modelo real |

<!-- Todo el documento es un plan de investigacion. Objetivo: falsar SpecWave. -->