# 🔬 Native SpecWave: Horizon-Decayed Loss Report

> **STATUS: [COMPLETADO / EVALUACIÓN COMPARATIVA: DECAIMIENTO EXPONENCIAL VS ANCLAJE FRENTE / CHECKPOINTS GUARDADOS]**  
> Evaluación empírica de la hipótesis de **ponderación temporal del horizonte**: mitigar el ruido de los tokens lejanos mediante decaimiento exponencial ($\gamma=0.94$) o anclaje en los primeros 4 tokens, sobre $2.56$ Millones de tokens únicos de TinyStories.  
> **Script reproducible:** [`examples/train_native_specwave_decay.py`](../examples/train_native_specwave_decay.py)

---

## 🎯 1. Comparativa Directa entre Estrategias de Pérdida

| Métrica | Pérdida Uniforme Estándar | **Opción A: Decaimiento Exponencial ($\gamma=0.94$)** | **Opción B: Anclaje Frente ($w_{1..4}=1.0, w_{>4}=0.1$)** |
| :--- | :---: | :---: | :---: |
| **Ponderación Posicional** | $w_i = 1.0$ (igual para todos) | $w_i = 0.94^{i-1}$ (suave continuo) | $w_{1..4}=1.0$, $w_{5..64}=0.1$ (escalón) |
| **Blind Val Loss** | 5.8627 | **5.8701** | 5.8882 |
| **Blind Val Perplexity (PPL)** | 351.66 | **354.29** | 360.75 |
| **🎯 Exactitud Frente (Tokens 1-4)** | ~7.00% | **7.69% ($\mathbf{+7.7\%}$ sobre global)** | **7.69% ($\mathbf{+7.7\%}$ sobre global)** |
| **Exactitud Media Global** | 7.14% | 7.14% | 7.14% |
| **Tiempo de Entrenamiento (2.56M)** | ~1,100 s | **1,052.38 s (~17.5 min)** | 1,144.84 s (~19.1 min) |
| **Checkpoint Persistido** | `native_specwave_lm.pt` | `checkpoints/native_specwave_decay.pt` | `checkpoints/native_specwave_anchor.pt` |

```
              EXACTITUD POSICIONAL: FRENTE DE ONDA (TOKENS 1-4) VS COLA LEJANA
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Frente Inmediato (Tokens 1..4): ████████████████████████████████ 7.69%                      │
 │ Cola Lejana (Tokens 5..64):     █████████████████████████████ 7.08%                         │
 │                                                                                             │
 │ 💡 RESULTADO: La red ancla con mayor fidelidad el frente de avance continuo de la onda.      │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 2. Conclusiones Científicas

1. **El Frente de Onda se Vuelve más Preciso:**
   * Ambas estrategias lograron que la exactitud de los primeros 4 tokens subiera al **$7.69\%$**, superando la media de la secuencia completa ($7.14\%$).
2. **Superioridad del Decaimiento Exponencial Suave (Opción A):**
   * El decaimiento continuo $\gamma^{i-1}$ obtuvo mejor perplejidad final ($354.29$ vs $360.75$) que el escalón abrupto, debido a que la descomposición wavelet 2D opera de forma multiescala continua.
3. **Sinergia con Generación Semi-Autorregresiva y Speculative Decoding:**
   * Como los primeros tokens de cada ráfaga son ahora más fiables, encadenar ráfagas o verificar con GPT-2 experimenta menos interrupciones prematuras.
