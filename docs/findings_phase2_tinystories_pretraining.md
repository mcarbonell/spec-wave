# 🚀 Phase 2 Benchmark Report: Pre-Training en 8 Historias Sintéticas (No TinyStories Real)

> **STATUS: [VALIDADO / MEMORIZACIÓN DE 8 PLANTILLAS / SPEEDUP MEDIDO SOBRE BASELINE PROPIO]**  
> Benchmark comparativo entre **SpecWave (Wave-In ➔ Wave-Out)** y un **mini-transformer causal propio** (llamado "Causal GPT-2 Baseline" en el script) sobre **8 plantillas de historias sintéticas hardcodeadas**, tokenizadas con el tokenizador BPE de GPT-2 ($V = 50,257$).  
> **Script reproducible:** [`examples/train_tinystories_specwave.py`](../examples/train_tinystories_specwave.py)  
> **Tarea:** Dado un prompt de 32 tokens, sintetizar el final de 32 tokens.

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Baseline Causal (Autoregresivo) | SpecWave (Wave-In ➔ Wave-Out) | **Ventaja SpecWave** |
| :--- | :---: | :---: | :---: |
| **Paradigma de Generación** | 32 pasos secuenciales | **1 solo paso ($O(1)$)** | **Single-Shot $O(1)$** |
| **Latencia de Generación (32 tokens)** | **$415.897\text{ ms}$** | **$8.271\text{ ms}$** | **$50.29\times$ FASTER** |
| **Loss Final de Pre-Training** | $0.0012$ | **$0.0007$** | Menor loss |
| **Perplexity (PPL) Final** | $1.0012$ | **$1.0007$** | Cerca de 1.0 |
| **Exact Ending Token Match (%)** | $100.00\%$ | **$100.00\%$** | $100.00\%$ (memorización) |
| **Convergencia (100% Match)** | Step 100 | **Step 50** | $2\times$ más rápida |

```
                 COMPARACIÓN DE LATENCIA DE GENERACIÓN (32 TOKENS)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Baseline Causal:  415.897 ms   ██████████████████████████████ (100.0%) │
 │ SpecWave O(1):      8.271 ms   █ (1.98% - 50.29x FASTER)               │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental Real

### 2.1 Dataset (IMPORTANTE: NO es TinyStories)

**El script NO descarga el dataset TinyStories de Eldan & Li (2023).** Los datos son **8 plantillas de historias hardcodeadas** en el código fuente (`examples/train_tinystories_specwave.py`, líneas 42–66), por ejemplo:

```python
("Once upon a time, Lily found a magical key in the garden. She unlocked the tiny wooden box and discovered a glowing blue bird.",
 "The blue bird sang a sweet song, and Lily smiled with joy. She knew she had made a wonderful new friend forever."),
```

**Procesamiento real** (líneas 68–108):
- Tokenización con `tiktoken` (encoding `gpt2`), truncando a 32 tokens por prompt y 32 por respuesta.
- Padding con token 0 hasta 32 tokens.
- **El "validation split" es un clon exacto del train** (líneas 105–106):
  ```python
  val_p = train_p.clone()
  val_r = train_r.clone()
  ```
- Resultado: **8 pares (prompt, respuesta) de 32 tokens cada uno**.

### 2.2 Arquitecturas Comparadas

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                           ARQUITECTURAS COMPARADAS (FASE 2)                                  │
 ├───────────────────────────────┬─────────────────────────────────────────────────────────────┤
 │           MODELO              │                        ESPECIFICACIÓN                        │
 ├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ A. Baseline Causal            │ TransformerEncoder de PyTorch: 4 capas, 4 cabezas,           │
 │    (llamado "GPT-2" en el     │ d_model=128, d_ffn=512, GELU, norm_first, máscara causal.   │
 │    script, pero NO es GPT-2)  │ SIN KV-cache: reprocesa toda la secuencia en cada paso.     │
 ├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ B. SpecWave Full Model        │ • 2D DWT Wavelet Prompt Encoder (Wave-In)                   │
 │                               │ • Reasoner MLP de 3 capas densas (frecuencia)               │
 │                               │ • 2D IDWT Parallel Language Vocoder (Wave-Out)              │
 └───────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

**Nota crítica:** El "Causal GPT-2 Baseline" es un `nn.TransformerEncoder` de PyTorch de 4 capas con `d_model=128`, **no** el GPT-2 de OpenAI. Además, su bucle autoregresivo **no usa KV-cache** (líneas 142–151): en cada paso reprocesa toda la secuencia acumulada, lo que infla artificialmente su latencia medida.

### 2.3 Hiperparámetros

```python
# Ambos modelos
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
# 301 pasos de entrenamiento
# Batch = 8 historias
# d_model = 128
```

---

## 🔬 3. Trayectoria de Pre-Training (301 Pasos)

```text
Step   | Baseline Loss | Baseline PPL | SpecWave Loss | SpecWave PPL  | SpecWave Match
------------------------------------------------------------------------------------------
Step 0   | 10.8655      | 52341.3233   | 11.0344       | 61966.6925    | 0.00          %
Step 50  | 0.0146       | 1.0147       | 0.0022        | 1.0022        | 100.00        % ──► CONVERGED
Step 100 | 0.0043       | 1.0043       | 0.0014        | 1.0014        | 100.00        %
Step 150 | 0.0027       | 1.0027       | 0.0011        | 1.0011        | 100.00        %
Step 200 | 0.0020       | 1.0020       | 0.0010        | 1.0010        | 100.00        %
Step 250 | 0.0015       | 1.0015       | 0.0008        | 1.0008        | 100.00        %
Step 300 | 0.0012       | 1.0012       | 0.0007        | 1.0007        | 100.00        % (Final)
```

---

## 🔍 4. Auditoría Cualitativa (Historia 1: Lily & El Pájaro Azul)

```text
[PROMPT (INICIO DE HISTORIA - 32 TOKENS)]:
"Once upon a time, Lily found a magical key in the garden. She unlocked the tiny wooden box 
and discovered a glowing blue bird."

[GROUND TRUTH TARGET ENDING (32 TOKENS)]:
"The blue bird sang a sweet song, and Lily smiled with joy. She knew she had made a 
wonderful new friend forever."

[SPECWAVE GENERATED ENDING (1 SOLO PASO O(1) EN 8.27 ms)]:
"The blue bird sang a sweet song, and Lily smiled with joy. She knew she had made a 
wonderful new friend forever."

[AUDIT RESULT]: 100.00% Exact Token Match.
```

---

## 💡 5. Interpretación Honesta de los Resultados

1. **El speedup de 50.29x es real pero sobre un baseline desventajado:** La latencia del baseline se mide (no es una constante), pero el baseline es un mini-transformer de 4 capas/d=128 **sin KV-cache**, que reprocesa toda la secuencia en cada uno de los 32 pasos. Un GPT-2 real con KV-cache sería sustancialmente más rápido, reduciendo el speedup real.

2. **No hay generalización demostrada:** El "validation split" es un clon del train. El 100.00% de exact match es **memorización de 8 plantillas**, no generación de historias nuevas. No se evalúa sobre historias no vistas.

3. **El nombre "TinyStories" es engañoso:** El dataset real de TinyStories contiene ~2.1M de historias. Aquí se usan 8 plantillas sintéticas.

4. **Conclusión:** La fase 2 demuestra que SpecWave puede memorizar 8 pares prompt→respuesta y que un forward único es más rápido que 32 forwards de un transformer sin KV-cache. Para validar la idea de verdad haría falta: (a) TinyStories real, (b) split train/test estricto, (c) baseline con KV-cache, y (d) métricas de generalización (PPL de test, coherencia evaluada por humanos o GPT-4).