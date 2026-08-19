# 🔬 Phase 1 Benchmark Report: Vocoder Invertibility on a Small Hardcoded Corpus

> **STATUS: [VALIDADO / RECONSTRUCCIÓN EXACTA SOBRE CORPUS LOCAL PEQUEÑO]**  
> Validación empírica del **Parallel 2D Wavelet Spectral Language Vocoder** sobre un corpus local de texto y código Python hardcodeado en el script, tokenizado con el tokenizador BPE oficial de GPT-2 ($V = 50,257$).  
> **Script reproducible:** [`tests/benchmark_vocoder_fineweb.py`](../tests/benchmark_vocoder_fineweb.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Medido | Nota |
| :--- | :---: | :--- |
| **Exact Token Match (%)** | **100.00%** | Sobre los bloques de entrenamiento (memorización) |
| **Reconstruction Perplexity (PPL)** | **1.0009** | Sobre los bloques de entrenamiento |
| **Cross-Entropy Loss** | **0.0009** | Sobre los bloques de entrenamiento |
| **Convergencia** | **50 pasos** | Corpus de ~12 bloques |
| **Vocab Scale** | **50,257 tokens (GPT-2 BPE)** | Tokenizador real |
| **Tiempo de Entrenamiento** | **49.33 segundos** (CPU) | — |

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                       TRAYECTORIA DE RECONSTRUCCIÓN (FASE 1)                                    │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ Step   0 (Init):  Loss = 10.9981  │  PPL = 59,760.2  │  Exact Match =   0.00%                   │
 │ Step  50 (Early): Loss =  0.0027  │  PPL =      1.0027  │  Exact Match = 100.00% ──► CONVERGED  │
 │ Step 100:         Loss =  0.0014  │  PPL =      1.0014  │  Exact Match = 100.00%                │
 │ Step 200:         Loss =  0.0010  │  PPL =      1.0010  │  Exact Match = 100.00%                │
 │ Step 300 (Final): Loss =  0.0009  │  PPL =      1.0009  │  Exact Match = 100.00%                │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental Real

### 2.1 Composición del Dataset (IMPORTANTE: corpus local, no WikiText-2)

**El script NO descarga WikiText-2 ni FineWeb.** Los datos son **dos strings literales embebidos en el código fuente** (`tests/benchmark_vocoder_fineweb.py`, líneas 46–80):

1. **Texto de muestra (etiquetado como "WikiText"):** Un párrafo de ~200 tokens sobre la teoría de la relatividad general de Einstein, copiado manualmente.
2. **Código Python de muestra:** Una implementación de `quick_sort` y una clase `PhasorMemoryMatrix` (~100 tokens), también hardcodeada.

**Procesamiento real** (líneas 82–105):
- Tokenización con `tiktoken` (encoding `gpt2`).
- Ventana deslizante con solapamiento: `range(0, len(raw) - seq_len, seq_len // 2)` → genera bloques de 64 tokens con 50% de solapamiento.
- **Duplicación de bloques** para rellenar el batch: `while len(blocks) % 4 != 0: blocks.append(blocks[0])` (línea 102).
- Resultado: **~12 bloques de 64 tokens, con solapamiento y duplicados**.

### 2.2 Arquitectura del Modelo

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                           ARQUITECTURA EXPERIMENTAL (FASE 1)                                 │
 ├───────────────────────────────┬─────────────────────────────────────────────────────────────┤
 │           LAYER               │                       SPECIFICATION                         │
 ├───────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ 1. Token Embeddings           │ Embedding(num_embeddings=50257, embedding_dim=128)          │
 │ 2. 2D Haar DWT Analysis       │ Matrix-free downsampling -> 4 subbands [B, 32, 64]          │
 │ 3. 2D Haar IDWT Synthesis     │ Matrix-free parallel upsampling -> [B, 64, 128]             │
 │ 4. Conv1D Spectral Refiner    │ Conv1d(128, 256, k=3, p=1) + GELU + Conv1d(256, 128, k=3)  │
 │ 5. Parallel De-quantizer Head │ Linear(in_features=128, out_features=50257, bias=False)    │
 └───────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### 2.3 Hiperparámetros de Entrenamiento

```python
optimizer = torch.optim.AdamW(
    params=list(embeddings.parameters()) + list(vocoder.parameters()),
    lr=4e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
num_steps = 300
batch_size = 12 bloques (768 tokens/paso)
loss_function = torch.nn.CrossEntropyLoss()
device = "cpu"
```

---

## 🔬 3. Dinámica de Entrenamiento Detallada

| Step | CrossEntropy Loss | Perplexity (PPL) | Exact Match (%) | Observaciones |
| :---: | :---: | :---: | :---: | :--- |
| **0** | $10.9981$ | $59,760.22$ | $0.00\%$ | Inicialización aleatoria sobre vocabulario de 50k. |
| **50** | $0.0027$ | $1.0027$ | **$100.00\%$** | Convergencia completa sobre los ~12 bloques. |
| **100** | $0.0014$ | $1.0014$ | **$100.00\%$** | — |
| **150** | $0.0011$ | $1.0011$ | **$100.00\%$** | — |
| **200** | $0.0010$ | $1.0010$ | **$100.00\%$** | — |
| **250** | $0.0009$ | $1.0009$ | **$100.00\%$** | — |
| **300** | **$0.0009$** | **$1.0009$** | **$100.00\%$** | Evaluación final. |

---

## 🔍 4. Auditoría Cualitativa de Reconstrucción

### Muestra 1: Texto académico (párrafo hardcodeado)
```text
[GROUND TRUTH]:
"...than two hundred years as a valid description of the gravitational force between 
masses. In Newton's model, gravity is the result of an attractive force between massive 
objects..."

[RECONSTRUCTED (1-STEP IDWT VOCODER)]:
"...than two hundred years as a valid description of the gravitational force between 
masses. In Newton's model, gravity is the result of an attractive force between massive 
objects..."

[MATCH]: 100.00% Exact Word-for-Word Match (0 errores).
```

### Muestra 2: Código Python (hardcodeado)
```python
# [GROUND TRUTH]:
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    ...

# [RECONSTRUCTED (1-STEP IDWT VOCODER)]:
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    ...

# [MATCH]: 100.00% Exact Indentation, Operator and Bracket Match.
```

---

## 💡 5. Interpretación Honesta de los Resultados

1. **El vocoder puede memorizar un corpus pequeño:** El resultado de 100.00% demuestra que un vocoder wavelet + refiner convolucional + head lineal puede reconstruir exactamente ~12 bloques de 64 tokens tras 50 pasos de entrenamiento. Esto valida la **capacidad de representación** (la transformada de Haar 2D es invertible y no pierde información), pero **no** demuestra generalización a texto no visto.

2. **Limitación crítica: no hay split de validación.** Todos los bloques (con solapamiento y duplicados) se usan tanto para entrenar como para evaluar. El PPL de 1.0009 es un PPL de **memorización**, no de generalización.

3. **El nombre del script (`benchmark_vocoder_fineweb.py`) es engañoso:** No se descarga ni FineWeb ni WikiText-2. El corpus son dos strings literales.

4. **Conclusión para fases posteriores:** La fase 1 valida la mecánica del vocoder (invertibilidad exacta de la transformada), pero el experimento de generalización real queda pendiente: requiere un corpus grande (p. ej., WikiText-2 real vía HuggingFace), un split train/test estricto y reportar PPL de validación.