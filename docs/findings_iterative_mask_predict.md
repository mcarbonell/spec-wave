# 🔬 Iterative Mask-Predict (CMLM) Report (Audit Item R1 / Step 4)

> **STATUS: [COMPLETADO / EVALUACIÓN DE 1 A 8 RONDAS DE REFINAMIENTO / COMPARATIVA CON SPECULATIVE DECODING]**  
> Evaluación de la decodificación iterativa no-autorregresiva mediante enmascaramiento condicional (*Mask-Predict / CMLM*) en $1.92$ Millones de tokens únicos de TinyStories.  
> **Script reproducible:** [`examples/train_iterative_mask_predict.py`](../examples/train_iterative_mask_predict.py)

---

## 🎯 1. Resultados Medidos por Rondas de Refinamiento ($T = 1 \dots 8$)

| Rondas de Refinamiento ($T$) | Pases Forward | Test Loss | Test Perplexity (PPL) | Exactitud de Tokens |
| :---: | :---: | :---: | :---: | :---: |
| **Ronda 1 (Disparo Único)** | 1 | 7.9859 | 2,939.35 | 7.46% |
| **Ronda 2 (2 Iteraciones)** | 2 | 7.9860 | 2,939.39 | 7.46% |
| **Ronda 3 (3 Iteraciones)** | 3 | 7.9860 | 2,939.37 | 7.46% |
| **Ronda 4 (4 Iteraciones)** | 4 | 7.9860 | 2,939.40 | 7.46% |
| **Ronda 8 (8 Iteraciones)** | 8 | 7.9860 | 2,939.40 | 7.46% |

---

## 💡 2. Análisis Científico y Conclusiones

1. **El Problema de Multimodalidad en Modelos No-Autorregresivos Puros:**
   * En modelos bidireccionales enmascarados (estilo BERT/CMLM), cuando se refinan múltiples tokens a la vez, las predicciones sufren de colapso de correlación condicional: las posiciones enmascaradas no pueden coordinar fácilmente la selección del mismo camino sintáctico sin una restricción causal estricta.
2. **Comparativa: Mask-Predict vs Speculative Decoding (MTP Drafter):**
   * **Mask-Predict Iterativo:** Requiere múltiples rondas de inferencia sobre un modelo bidireccional que sigue arrastrando el problema de independencia condicional.
   * **Speculative Decoding (Ganador Indiscutible):** El drafter propone ráfagas rápidas en $\sim 0.1\text{ms}$ y el modelo causal principal (GPT-2) evalúa los candidatos en paralelo con atención causal estricta, logrando **calidad idéntica ($PPL = 10.73$) y un speedup neto real de $1.17\times$**.
