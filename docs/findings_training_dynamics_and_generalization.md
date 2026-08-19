# 🔬 Empirical Report: Training Dynamics en WikiText-2 Real — Memorización sin Generalización

> **STATUS: [VALIDADO PARCIALMENTE / DATOS REALES / TRAIN PPL 1.02 / VAL PPL ~4,500 (NO GENERALIZA)]**  
> Estudio de la dinámica de optimización de un adaptador SpecWave sobre **WikiText-2 real** (descargado de HuggingFace) con GPT-2 real congelado. El entrenamiento converge a PPL 1.02 en train, pero la **validación blind se estanca en PPL ~4,500**, demostrando que el modelo memoriza sin generalizar.  
> **Script reproducible:** [`examples/benchmark_ppl_parity.py`](../examples/benchmark_ppl_parity.py)  
> **Notebook Colab:** [`examples/specwave_gpt2_colab_demo.ipynb`](../examples/specwave_gpt2_colab_demo.ipynb)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Medido | Interpretación Honesta |
| :--- | :---: | :--- |
| **Dataset** | **WikiText-2 real** (HuggingFace) | ✅ Datos reales |
| **Muestras de entrenamiento** | 600 pares (prompt 64 → target 64) | Corpus pequeño |
| **Muestras de validación** | 100 pares blind (test split) | Split estricto |
| **Train PPL final** | **1.02** | Memorización del train |
| **Val PPL final (blind)** | **~4,482** (loss ~8.4) | **NO generaliza** |
| **Baseline GPT-2 nativo** | PPL ~25-35 (medido en el script) | Referencia real |
| **Hardware** | Tesla T4 (Colab) | GPU real |

```
                       DINÁMICA DE LOSS (STEPS 0 A 900 EN T4 GPU)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Step 0:   Loss 17.30 (PPL 32,823,519.8)  ██████████████████████████████████████████ (100.0%) │
 │ Step 200: Loss  6.33 (PPL 562.03)        ██████████████▍ (36.5%)                            │
 │ Step 500: Loss  4.65 (PPL 105.47)        ██████████▋ (26.8%)                                │
 │ Step 580: Loss  2.65 (PPL 14.16)         ██████ (15.3% - "PHASE TRANSITION")                │
 │ Step 660: Loss  0.11 (PPL 1.13)          ▍ (0.6% - TRAIN NEAR-LOSSLESS)                     │
 │ Step 800: Loss  0.02 (PPL 1.02)          ▏ (0.1% - TRAIN MEMORIZATION)                      │
 │                                                                                             │
 │ ⚠️  VALIDACIÓN BLIND: Loss ~8.4 (PPL ~4,500) — EL MODELO NO GENERALIZA                     │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental Real

### 2.1 Dataset (SÍ es real)

A diferencia de las fases 1-3, este experimento **sí descarga WikiText-2 real** desde HuggingFace (`examples/benchmark_ppl_parity.py`, líneas 210–226):

```python
raw_train = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train")
raw_test = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")
```

- 600 pares de entrenamiento (prompt 64 tokens → target 64 tokens).
- 100 pares de validación blind (test split).
- Tokenizador GPT-2 real.

### 2.2 Arquitectura

- **GPT-2 real (124M)** cargado desde HuggingFace.
- Capas 0-10 **congeladas**; capa 11 + LayerNorm final **descongeladas** para co-adaptación.
- **LM Head con weight-tying**: se copian los pesos preentrenados de `lm_head` de GPT-2.
- **Spectral Reasoner:** MLP de 2 capas (64×768 → 64×768).
- **Refiner:** 2 Conv1D residuales + LayerNorm.
- **Loss híbrida:** CrossEntropy + 2.0 × MSE(embeddings reconstruidos, embeddings target).

### 2.3 Entrenamiento

```python
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# num_steps = 100 por defecto (el doc reporta 900)
# batch_size = 4
```

---

## 🔬 3. Dinámica de Entrenamiento Reportada (900 pasos)

| Epoch | Train Loss | Train PPL | Estado Cualitativo |
| :--- | :---: | :---: | :--- |
| **Inicial (Step 0)** | $17.3067$ | $32,823,519.81$ | Inicialización aleatoria |
| **Exploración (Step 200)** | $6.3316$ | $562.03$ | Alineación macro-sintáctica |
| **Pre-Transición (Step 500)** | $4.6584$ | $105.47$ | Estabilización armónica |
| **"Phase Shift" (Step 580)** | $2.6501$ | $14.16$ | Descenso brusco de loss |
| **Convergencia train (Step 800)** | $0.0239$ | $1.0200$ | Memorización del train |

### Validación Blind (el dato que el doc original minimiza)

El propio doc original admite:

> "out-of-distribution validation across 100 unseen blind test articles stabilized at **Loss ≈ 8.65**"

Eso equivale a **PPL ≈ 5,700**. El script `benchmark_streaming_generalization.py` reporta val PPL ~4,482 (loss ~8.4). En ambos casos, la validación **no baja de loss ~8.4**, mientras que el train llega a loss 0.02.

---

## 💡 4. Interpretación Honesta de los Resultados

1. **El "phase transition" es solo el descenso de loss de memorización:** La caída brusca de loss en el paso 580 es el momento en que el modelo empieza a memorizar los 600 pares de entrenamiento. No es una "transición de fase" científica ni un fenómeno nuevo: es el comportamiento típico de un modelo con capacidad suficiente para memorizar un dataset pequeño.

2. **El modelo NO generaliza:** Train PPL 1.02 vs Val PPL ~4,500-5,700. La brecha de 4 órdenes de magnitud entre train y val es la definición de overfitting. El doc original lo presentaba como "dos regímenes" (closed-domain vs open-domain), pero la interpretación correcta es que el adaptador memoriza los pares de entrenamiento y no aprende a predecir texto nuevo.

3. **Lo genuino del experimento:** Es el único benchmark del repo con datos reales (WikiText-2), split estricto, baseline real (GPT-2 nativo medido) y hardware GPU real (T4). La metodología es la correcta; el resultado es negativo para la hipótesis de generalización.

4. **Conclusión:** Este experimento demuestra que el vocoder SpecWave puede memorizar 600 pares de WikiText-2 (train PPL 1.02) pero **no generaliza** a texto no visto (val PPL ~4,500). Para que SpecWave fuera viable como modelo de lenguaje haría falta: (a) entrenar con datasets mucho más grandes (N > 50,000 como sugiere el propio doc), (b) evaluar PPL de validación como métrica principal, y (c) comparar contra GPT-2 real en las mismas condiciones.