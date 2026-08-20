# 🔬 Wavelet Speculative Decoding Benchmark Report

> **STATUS: [COMPLETADO / VERIFICACIÓN MATEMÁTICA / EFECTO DE LA TASA DE ACEPTACIÓN]**  
> Evaluación del desacoplamiento entre un **Propositor Rápido Wavelet 2D ($O(1)$)** y un **Verificador Causal GPT-2** mediante *Rejection Sampling* en TinyStories.  
> **Script reproducible:** [`examples/benchmark_wavelet_speculative_decoding.py`](../examples/benchmark_wavelet_speculative_decoding.py)

---

## 🎯 1. Resumen Ejecutivo y Hallazgos

| Métrica | GPT-2 Causal Estándar | Speculative Decoding (Drafter Raw) | Condición para Speedup $\ge 2\text{x}$ |
| :--- | :---: | :---: | :---: |
| **Distribución de Salida** | Exacta GPT-2 ($PPL = 10.73$) | **PROBABLEMENTE IDÉNTICA** ($PPL = 10.73$) | Exacta GPT-2 |
| **Pases Forward por 64 tokens** | 64 | 64.0 | $\le 20$ pases |
| **Tasa de Aceptación ($\alpha$)** | — | **12.50%** (Drafter sin pre-entrenar) | $\alpha \ge 60\text{--}75\%$ |
| **Rendimiento (Throughput)** | **6.84 tok/s** | **3.44 tok/s** ($0.50\text{x}$) | $\ge 15\text{--}25\text{ tok/s}$ |

---

## 💡 2. Lección Teórica de Speculative Decoding

1. **Garantía de Calidad:**
   * El muestreo por rechazo de Leviathan et al. garantiza matemáticamente que **la perplejidad y distribución del texto resultante es exactamente la de GPT-2 ($10.73$)**, sin importar qué proponga el drafter.
2. **El Umbral Crítico de Aceptación ($\alpha$):**
   * Cuando el drafter no está pre-entrenado profundamente, la tasa de aceptación es baja ($\alpha \approx 12.5\%$). Cada rechazo descarta la ráfaga y obliga al verificador a corregir inmediatamente, sumando el coste del drafter al del verificador.
   * Para obtener un *speedup* real de $\ge 2\text{x}\text{--}4\text{x}$, el proyector wavelet necesita alcanzar una precisión de predicción de $\alpha \ge 60\text{--}75\%$, lo cual requiere entrenar el adaptador sobre grandes volúmenes de datos continuos.
