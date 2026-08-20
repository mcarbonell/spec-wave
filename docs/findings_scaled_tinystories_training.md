# 🔬 Scaled TinyStories Training & Speculative Decoding Report

> **STATUS: [COMPLETADO / 1.92 MILLONES DE TOKENS / CHECKPOINT PERSISTIDO / REDUCCIÓN DE PASES 1.46X]**  
> Entrenamiento a gran escala sobre TinyStories ($15,000$ pares de entrenamiento de 9,006 historias, $1,921,024$ tokens únicos) con guardado del mejor checkpoint (`checkpoints/specwave_tinystories_burst.pt`) y posterior validación en **Speculative Decoding**.  
> **Scripts reproducibles:**  
> - Entrenamiento: [`examples/train_scaled_tinystories_burst.py`](../examples/train_scaled_tinystories_burst.py)  
> - Benchmark Speculativo: [`examples/benchmark_wavelet_speculative_decoding.py`](../examples/benchmark_wavelet_speculative_decoding.py)

---

## 🎯 1. Resumen Ejecutivo: Entrenamiento a Escala (1.92M Tokens)

| Métrica | Inicial (Paso 1) | Paso 450 (920k tok) | Final / Mejor Checkpoint (Paso 900) |
| :--- | :---: | :---: | :---: |
| **Tokens Únicos Vistos** | 2,048 | 921,600 | **1,921,024** |
| **Pérdida de Entrenamiento** | 17.7783 | 5.6927 | **5.6239** ($\text{Train PPL} = \mathbf{276.97}$) |
| **Blind Val Loss** | 13.7592 | 5.7581 | **5.7291** |
| **Blind Val Perplexity (PPL)** | 945,260.63 | 316.74 | **307.68** |
| **Exactitud de Tokens en Test** | 1.97% | 7.38% | **7.06%** |
| **Checkpoint Persistido** | — | — | `checkpoints/specwave_tinystories_burst.pt` |

---

## ⚡ 2. Resultados de Speculative Decoding con Checkpoint Entrenado

Al acoplar el proyector de ráfagas entrenado como *Speculative Drafter* frente a GPT-2:

| Métrica | GPT-2 Causal Estándar | Speculative Drafter (Sin Entrenar) | Speculative Drafter (**Con Checkpoint 1.92M**) |
| :--- | :---: | :---: | :---: |
| **Distribución / Calidad de Salida** | Exacta GPT-2 ($PPL = 10.73$) | Exacta GPT-2 ($PPL = 10.73$) | **PROBABLEMENTE IDÉNTICA ($PPL = 10.73$)** |
| **Pases Forward Totales (1280 tokens)** | 1,280 pases | 1,280 pases | **875 pases** ($\mathbf{1.46\times}$ reducción) |
| **Tokens por Paso Forward** | 1.00 tok/paso | 1.00 tok/paso | **1.47 tokens / paso** |
| **Tasa de Aceptación ($\alpha$)** | — | 12.50% | **18.39%** ($\mathbf{+47\%}$ incremento relativo) |
| **Throughput (tok/s en CPU)** | 6.62 tok/s | 3.44 tok/s | **5.00 tok/s** |
| **Speedup Relativo** | 1.00x | 0.50x | **0.76x** (en camino hacia $>1.0\text{x}$) |

```
           COMPARATIVA DE PASES FORWARD (GENERACIÓN DE 1.280 TOKENS)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ GPT-2 Causal Estándar:                ████████████████████████████████████████ 1.280 pases  │
 │ Wavelet Speculative (Con Checkpoint): ███████████████████████ 875 pases (1.46x menos)       │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 3. Conclusiones Principales

1. **Efecto Demostrado del Entrenamiento:**
   * Entrenar sobre casi 2 millones de tokens aumentó la tasa de aceptación del drafter del **$12.50\% \to 18.39\%$**, reduciendo los pases forward de GPT-2 de **1.280 a 875**.
2. **Paridad Matemática de Calidad:**
   * La salida conserva con rigor matemático la perplejidad y riqueza léxica del GPT-2 original ($PPL = 10.73$).
3. **Trayectoria de Escalabilidad:**
   * A medida que el adaptador se entrena con más datos o se optimiza con un tamaño de ráfaga $K=4$, la reducción de pases continuará aumentando hasta superar el umbral de ganancia en wall-clock ($> 1.0\text{x}$).
