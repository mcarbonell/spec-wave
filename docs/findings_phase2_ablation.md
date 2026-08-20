# 🔬 Phase 2 Benchmark Report: Wavelet Ablation Study (SpecWave vs. Flat Baseline)

> **STATUS: [COMPLETADO / RESULTADO CIENTÍFICO CLAVE / EQUIVALENCIA B ≈ A]**  
> Estudio comparativo riguroso de ablación para aislar el impacto inductivo de la **Transformada Wavelet 2D de Haar** frente a una **representación plana (Flat Baseline)** con el mismo número exacto de parámetros ($21,723,008$) sobre pares de secuencias reales de WikiText-2 (Prompt 64 tokens $\to$ Target 64 tokens).  
> **Script reproducible:** [`benchmarks/phase2_ablation.py`](../benchmarks/phase2_ablation.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Modelo A: SpecWave (Wavelets 2D) | Modelo B: Flat Baseline (Sin Wavelets) | Diferencia ($\Delta = B - A$) |
| :--- | :---: | :---: | :---: |
| **Parámetros del Modelo** | **21,723,008** | **21,723,008** | **0** (idéntico) |
| **Blind Val Loss** | **6.9927** | **6.9926** | **-0.0001** |
| **Blind Val Perplexity (PPL)** | **1088.70** | **1088.57** | **-0.13** ($< 0.01\%$) |
| **Blind Val Token Accuracy** | **5.03%** | **5.03%** | **0.00%** |
| **Tiempo de Entrenamiento** | **140.68 s** | **143.53 s** | **+2.85 s** |

```
                       COMPARACIÓN DE ABLACIÓN (FASE 2)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Modelo A (SpecWave 2D DWT): Val Loss = 6.9927 │ Val PPL = 1088.70 │ Val Acc = 5.03%         │
 │ Modelo B (Flat Baseline):   Val Loss = 6.9926 │ Val PPL = 1088.57 │ Val Acc = 5.03%         │
 │                                                                                             │
 │ ⚠️  VEREDICTO GATE 2: EQUIVALENCIA FUNCIONAL EXACTA (B ≈ A)                                  │
 │     La transformada de Haar 2D es una rotación ortogonal lineal en el espacio de Hilbert.   │
 │     Las capas densas (MLP) aprenden un subespacio isomórfico con o sin descomposición Haar. │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 2. Protocolo Experimental

### 2.1 Dataset y Particiones
- **Fuente:** WikiText-2 real tokenizado con GPT-2 BPE ($V=50,257$).
- **Tarea:** Predicción condicionada de bloque siguiente (Prompt 64 tokens $\to$ Target 64 tokens).
- **Partición Train:** 4,000 pares de secuencias (512,000 tokens).
- **Partición Test Blind:** 500 pares de secuencias independientes (64,000 tokens).

### 2.2 Comparativa Arquitectónica
- **Modelo A (SpecWave):** $\mathbf{X}_{\text{prompt}} \to \text{Embed} \to \text{2D DWT} \to \text{MLP Resonante} \to \text{2D IDWT} \to \text{Conv Refiner} \to \text{LM Head}$.
- **Modelo B (Flat Baseline):** $\mathbf{X}_{\text{prompt}} \to \text{Embed} \to \text{MLP Espacial Plano} \to \text{Conv Refiner} \to \text{LM Head}$.

Ambos modelos utilizan idéntica dimensionalidad ($d_{\text{model}}=128$, hidden dim $512$), mismo optimizador (`AdamW`, $lr=2\times 10^{-3}$, `CosineAnnealingLR`) y mismo hardware (CPU Zen 4).

---

## 🔬 3. Dinámica de Entrenamiento Detallada

### Modelo A: SpecWave (Wavelets 2D)
- **Paso 1 (Init):** Train Loss = $11.0132$ | Train PPL = $60,671.21$ | Val Loss = $10.7219$ | Val PPL = $45,340.09$ | Val Acc = $0.16\%$
- **Paso 100:** Train Loss = $6.7019$ | Train PPL = $813.93$ | Val Loss = $6.9956$ | Val PPL = $1,091.81$ | Val Acc = $5.03\%$
- **Final (Split Test Completo):** Blind Val Loss = **6.9927** | Blind Val PPL = **1088.70** | Val Acc = **5.03%**

### Modelo B: Flat Baseline (Sin Wavelets)
- **Paso 1 (Init):** Train Loss = $11.0080$ | Train PPL = $60,353.64$ | Val Loss = $10.7146$ | Val PPL = $45,007.71$ | Val Acc = $0.12\%$
- **Paso 100:** Train Loss = $6.7019$ | Train PPL = $813.92$ | Val Loss = $6.9954$ | Val PPL = $1,091.57$ | Val Acc = $5.03\%$
- **Final (Split Test Completo):** Blind Val Loss = **6.9926** | Blind Val PPL = **1088.57** | Val Acc = **5.03%**

---

## 💡 4. Conclusiones Científicas de la Fase 2

1. **Equivalencia de Representación:** La transformada discreta de Haar 2D es un operador lineal ortogonal invertible (preserva normas por Parseval). Al acoplarse a capas densas $\mathbf{W}$, la composición $\mathbf{W} \cdot \mathbf{DWT}$ es matemáticamente una reparametrización lineal de $\mathbf{W}_{\text{flat}}$. Empíricamente, la red aprende exactamente al mismo ritmo con y sin wavelets ($\Delta \text{Loss} = 0.0001$).
2. **El Problema Real de la Generación NAR de 1 Paso:** Ambos modelos alcanzan un Val PPL de $\approx 1088$ en generación condicionada determinista de un solo paso. Esto confirma la hipótesis del roadmap: el cuello de botella no está en si se usan wavelets o vectores planos, sino en que **un mapeo directo $N \times d \to N \times d$ no puede modelar la distribución multimodal del lenguaje en un solo disparo**.
3. **Implicación para la Fase 3:** La única vía no-autorregresiva con potencial de modelar distribuciones complejas continuas es el **refinamiento iterativo / difusor (Diffusion-LM en el espacio latente continuo/espectral)**.
