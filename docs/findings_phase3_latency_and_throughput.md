# ⚡ Phase 3 Benchmark Report: Latencia del Vocoder y Throughput (Speedups Parcialmente Extrapolados)

> **STATUS: [VALIDADO PARCIALMENTE / LATENCIA SPECWAVE MEDIDA / BASELINE EXTRAPOLADO PARA N≥128]**  
> Perfilado de latencia del vocoder SpecWave para bloques de $N \in [32, 64, 128, 256]$ tokens y throughput de serving concurrente ($\text{Batch} \in [1, 4, 16, 64]$) usando el vocabulario GPT-2 ($V = 50,257$).  
> **Script reproducible:** [`tests/benchmark_gpu_wallclock.py`](../tests/benchmark_gpu_wallclock.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Reportado | Cómo se obtuvo |
| :--- | :---: | :--- |
| **Speedup máximo ($N=256$)** | **$155.56\times$** | **Baseline EXTRAPOLADO con fórmula** (no medido) |
| **Latencia SpecWave $N=32$** | **$7.33\text{ ms}$** | Medida (vocoder con entrada aleatoria) |
| **Latencia SpecWave $N=64$** | **$10.19\text{ ms}$** | Medida (vocoder con entrada aleatoria) |
| **Latencia SpecWave $N=128$** | **$18.20\text{ ms}$** | Medida (vocoder con entrada aleatoria) |
| **Latencia SpecWave $N=256$** | **$27.71\text{ ms}$** | Medida (vocoder con entrada aleatoria) |
| **Peak Throughput (64 usuarios)** | **$13,589.74\text{ tokens/sec}$** | Medido sobre vocoder **sin entrenar** con entrada aleatoria |

```
                 LATENCIA DE GENERACIÓN EN FUNCIÓN DE N
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ N = 32 Tokens:   Baseline:  272.16 ms   ██████████████                               │
 │                  Spec:        7.33 ms   ▏ (37.1x FASTER)                             │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ N = 64 Tokens:   Baseline:  630.88 ms   ██████████████████████████                   │
 │                  Spec:       10.19 ms   ▏ (61.9x FASTER)                             │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ N = 128 Tokens:  Baseline: 1909.76 ms   ████████████████████████████████████████████ │
 │                  Spec:       18.20 ms   ▏ (104.9x FASTER)  ← Baseline EXTRAPOLADO    │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ N = 256 Tokens:  Baseline: 4311.04 ms   █████████████████████████████████████████████ │
 │                  Spec:       27.71 ms   ▏ (155.6x FASTER) ← Baseline EXTRAPOLADO     │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 2. Protocolo Experimental Real

### 2.1 Medición de Latencia SpecWave (medida real)

Para cada $N$, se crea un `SpecWaveLanguageModel` con `seq_len=N` y se mide `single_shot_generate` sobre un vector de pensamiento **aleatorio** (`torch.randn(1, d_model)`):

```python
thought_vec = torch.randn(1, d_model, device=device)   # línea 102
spec_ms = ((time.perf_counter() - t0) / iters) * 1000.0  # 10 iteraciones
```

**Nota:** La entrada es ruido aleatorio. No hay datos, ni prompt, ni generación de texto real. Solo se mide el coste de cómputo del vocoder.

### 2.2 Medición del Baseline (parcialmente fabricada)

**Para $N \le 64$, el baseline SÍ se mide** con un `LightweightCausalTransformer` (4 capas, d=128, sin KV-cache):

```python
if n <= 64:
    gpt_ms = ...  # medición real con generate_n_tokens()
else:
    # Empirical linear-quadratic projection for N=128, 256 on CPU
    base_rate = 13.0  # ms per token on CPU          # ← CONSTANTE INVENTADA
    gpt_ms = n * base_rate + (n ** 2) * 0.015        # ← FÓRMULA, NO MEDICIÓN
```

**Para $N = 128$ y $N = 256$, el baseline NO se mide:** se extrapola con la fórmula `gpt_ms = n * 13.0 + n² * 0.015`. Los speedups de **104.95x** y **155.56x** dependen completamente de esta constante arbitraria.

### 2.3 Medición de Throughput (medida real, pero sin datos)

```python
batch_thoughts = torch.randn(num_users, d_model, device=device)  # entrada aleatoria
toks_per_sec = (total_tokens / (batch_ms / 1000.0))
```

El throughput de 13,589.7 tok/s se mide sobre un vocoder **sin entrenar** procesando **ruido aleatorio** en batch. No hay generación de texto, ni modelo de lenguaje, ni datos reales.

---

## 📊 3. Tabla de Resultados Reportados

| Block Length ($N$) | Baseline Latencia | SpecWave Latencia | Speedup | Método Baseline |
| :---: | :---: | :---: | :---: | :--- |
| **$N = 32$** | $272.16\text{ ms}$ | **$7.33\text{ ms}$** | **$37.14\times$** | Medido |
| **$N = 64$** | $630.88\text{ ms}$ | **$10.19\text{ ms}$** | **$61.90\times$** | Medido |
| **$N = 128$** | $1,909.76\text{ ms}$ | **$18.20\text{ ms}$** | **$104.95\times$** | **EXTRAPOLADO** |
| **$N = 256$** | $4,311.04\text{ ms}$ | **$27.71\text{ ms}$** | **$155.56\times$** | **EXTRAPOLADO** |

### Throughput Concurrente (N=64 tokens)

| Concurrent Users | Total Batch Latency | Throughput (Tokens/sec) | Requests / Sec |
| :---: | :---: | :---: | :---: |
| **1 User** | $10.24\text{ ms}$ | $6,252.94\text{ tok/s}$ | $97.70\text{ req/s}$ |
| **4 Users** | $23.86\text{ ms}$ | $10,729.21\text{ tok/s}$ | $167.64\text{ req/s}$ |
| **16 Users** | $102.73\text{ ms}$ | $9,967.71\text{ tok/s}$ | $155.75\text{ req/s}$ |
| **64 Users** | $301.40\text{ ms}$ | **$13,589.74\text{ tok/s}$** | **$212.34\text{ req/s}$** |

---

## 💡 4. Interpretación Honesta de los Resultados

1. **La latencia de SpecWave es real y baja:** El vocoder es ligero y un forward único es rápido. Esto es genuino y esperable: un MLP + IDWT + head lineal es computacionalmente barato.

2. **Los speedups de 104.95x y 155.56x NO son mediciones:** Dependen de una constante arbitraria (`base_rate = 13.0 ms/token`) y una fórmula cuadrática inventada. Cambiar esa constante cambia el speedup. Además, el baseline medido (para N≤64) es un transformer de 4 capas/d=128 **sin KV-cache**, no un LLM real.

3. **El throughput no representa un sistema de lenguaje:** Se mide sobre ruido aleatorio con un vocoder sin entrenar. No hay tokens generados, ni modelo de lenguaje, ni usuarios reales.

4. **Conclusión:** La fase 3 demuestra que el vocoder SpecWave es computacionalmente barato (latencia sub-30ms para N=256). Pero los claims de "155.56x" y "13,589 tok/s" como speedup frente a LLMs autoregresivos reales **no están respaldados por mediciones**. Para validarlos haría falta: (a) medir GPT-2 real con KV-cache en el mismo hardware, (b) medir en GPU (el script se ejecuta en CPU si no hay CUDA), y (c) generar texto real, no ruido.