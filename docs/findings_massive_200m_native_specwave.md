# 🔬 Massive 200 Million Tokens Native SpecWave LM Report

> **STATUS: [COMPLETADO / 200,003,584 TOKENS ÚNICOS / 20.2 HORAS DE ENTRENAMIENTO OVERNIGHT / CHECKPOINTS PERSISTIDOS]**  
> Entrenamiento a escala ultra-masiva de la arquitectura nativa **Native SpecWave LM** sobre **200 Millones de Tokens Únicos** de TinyStories con pérdida de Decaimiento Exponencial del Horizonte ($\gamma=0.94$).  
> **Script reproducible:** [`examples/train_native_specwave_decay.py`](../examples/train_native_specwave_decay.py)

---

## 🎯 1. Resumen Ejecutivo y Estadísticas del Entrenamiento

| Métrica | Valor Medido | Observaciones |
| :--- | :---: | :--- |
| **Tokens Únicos Procesados** | **200,003,584** | ~1.56 Millones de pares $100\%$ sin repetición |
| **Tiempo Total de Cómputo** | **72,675.58 s (~20.18 horas)** | Ejecución nocturna completa en CPU AMD Ryzen 7 8845HS |
| **Velocidad Media Sostenida** | **~2,752 tokens/segundo** | Alto rendimiento AVX-512 continuo |
| **Mejor Pérdida de Validación** | **5.9122** | Registrado en el paso 36.000 (~147M tokens) |
| **Mejor Perplejidad en Test Ciego** | **369.51** | Estable y sin sobreajuste a lo largo de 200M tokens |
| **🎯 Exactitud Frente (Tokens 1-4)** | **7.69%** | Anclaje de frente consistente |
| **Exactitud Global de Tokens** | **7.14%** | Media a lo largo de los 64 tokens |
| **Checkpoints Persistidos** | `checkpoints/native_specwave_decay_200m.pt`<br>+ 15 checkpoints periódicos (`_step*.pt`) | Totalmente guardados |

```
              TRAYECTORIA DE VALIDACIÓN EN 200 MILLONES DE TOKENS
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Paso 1      (4k tokens):    Val Loss = 10.7223 │ Val PPL = 45,356.50                        │
 │ Paso 5,000  (20M tokens):   Val Loss =  5.9412 │ Val PPL =    380.39                        │
 │ Paso 15,000 (60M tokens):   Val Loss =  5.9230 │ Val PPL =    373.50                        │
 │ Paso 25,000 (100M tokens):  Val Loss =  5.9129 │ Val PPL =    369.78                        │
 │ Paso 36,000 (147M tokens):  Val Loss =  5.9122 │ Val PPL =    369.51 ──► BEST GLOBAL CKPT   │
 │ Paso 48,500 (200M tokens):  Val Loss =  5.9128 │ Val PPL =    369.76                        │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 2. Hallazgos Científicos y Leyes de Capacidad

1. **Estabilidad Absoluta y Cero Degeneración:**
   * El modelo entrenó durante más de **20 horas continuas y 200 millones de tokens** sin una sola inestabilidad numérica, explosión de gradientes ni colapso de representación.
2. **Meseta de Capacidad de Parámetros (~34M params):**
   * El modelo converge rápidamente hasta los **~40-60M tokens**, momento en el cual la arquitectura de $34.1\text{M}$ parámetros ($d_{\text{model}}=384$, 6 capas) alcanza su **límite de capacidad expresiva** para la predicción de disparo único de 64 tokens, estabilizándose en $\text{Loss} \approx 5.91$ ($\text{PPL} \approx 369$).
3. **Confirmación del Límite de Disparo Único vs Ráfagas:**
   * Entrenar con más tokens (de 2M a 200M) estabiliza y robustece el modelo, pero confirma que para dar el salto a PPLs de dos dígitos ($< 30$), la vía más eficaz es la **generación en ráfagas semi-autorregresivas $O(4)$** o el **Speculative Decoding**.
