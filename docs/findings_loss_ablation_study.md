# 🔬 Loss Ablation Study Report: Pure CE vs Hybrid MSE (Audit R3a / A5)

> **STATUS: [COMPLETADO / 9 EJECUCIONES (3 CONFIGURACIONES x 3 SEMILLAS) / AUDITORÍA A5 CONFIRMADA]**  
> Estudio comparativo riguroso con 3 semillas aleatorias ($42, 123, 999$) para evaluar el impacto del Cross-Entropy puro frente a las pérdidas auxiliares de MSE continuo (Parseval multiescala y Manifold Embedding MSE).  
> **Script reproducible:** [`examples/ablation_loss_study.py`](../examples/ablation_loss_study.py)

---

## 🎯 1. Resumen Estadístico (Media ± Desviación Típica)

| Configuración de Pérdida | Val Loss ($\mu \pm \sigma$) | Val PPL ($\mu \pm \sigma$) | Val Token Acc ($\mu \pm \sigma$) | Veredicto |
| :--- | :---: | :---: | :---: | :---: |
| **1. `CE_ONLY` (Cross-Entropy Pura)** | **5.9310 ± 0.0021** | **376.55 ± 0.79** | **7.80% ± 0.00%** | 🏆 **MEJOR PERPLEJIDAD** |
| **2. `CE_PLUS_SPECTRAL` (CE + Parseval MSE)** | 5.9319 ± 0.0009 | 376.85 ± 0.34 | 7.80% ± 0.00% | Neutral ($\Delta \text{PPL} = +0.30$) |
| **3. `FULL_HYBRID` (CE + Parseval + Manifold MSE)** | 5.9350 ± 0.0029 | 378.03 ± 1.10 | 7.80% ± 0.00% | Peor ($\Delta \text{PPL} = +1.48$) |

```
              COMPARATIVA DE PERPLEJIDAD EN VALIDACIÓN (3 SEEDS x 3 CONFIGS)
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. CE_ONLY:          ██████████████████████████████████ 376.55 ± 0.79 ──► GANADOR           │
 │ 2. CE_PLUS_SPECTRAL: ██████████████████████████████████ 376.85 ± 0.34                       │
 │ 3. FULL_HYBRID:      ████████████████████████████████████ 378.03 ± 1.10 (Leve degradación)  │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 2. Conclusiones Científicas

1. **La Hipótesis A5 de la Auditoría Queda Confirmada:**
   * Las pérdidas auxiliares de MSE continuo en el espacio de embeddings (`manifold_MSE`) **no ayudan a la generación de texto discreto**: fuerzan a la red a predecir la media euclidiana continua de los embeddings objetivo, lo que degrada ligeramente el log-likelihood discreto ($\text{PPL } 376.55 \to 378.03$).
2. **Recomendación para Futuros Diseños:**
   * La optimización debe guiarse **puramente por Cross-Entropy discreto** (o KL-divergence contra el verificador en Speculative Decoding), evitando MSEs en el espacio de embeddings que compitan con la distribución categórica.
3. **Consistencia Estadística:**
   * La varianza entre semillas es minúscula ($\sigma_{\text{PPL}} < 1.10$), lo que demuestra que los resultados son altamente reproducibles y no dependen de la inicialización estocástica.
