# 🛡️ Empirical Report: Phase 4 — Clasificador sobre Ruido Gaussiano Sintético (No Safety Real)

> **STATUS: [VALIDADO COMO DEMO / 100% SOBRE DATOS SINTÉTICOS TRIVIALMENTE SEPARABLES]**  
> Validación de un clasificador ligero ("tripwire") que distingue dos nubes de ruido gaussiano sintético con medias opuestas, descompuestas en subbandas wavelet 2D. **No involucra texto, jailbreaks, ni intenciones reales.**  
> **Script reproducible:** [`tests/benchmark_spectral_safety.py`](../tests/benchmark_spectral_safety.py)

---

## 🎯 1. Resumen Ejecutivo y Resultados Medidos

| Métrica | Resultado Reportado | Qué significa realmente |
| :--- | :---: | :--- |
| **Detección de ataques** | **$100.00\%$ (50/50)** | Clasificación de ruido gaussiano con media -1.2 vs +1.2 |
| **Falsos positivos** | **$0.00\%$ (50/50)** | Idem |
| **Latencia total de auditoría** | **$0.0937\text{ ms}$** | Coste de un DWT + MLP de 2 capas sobre tensores sintéticos |
| **Energía en subbanda LL** | **$93.51\%$** | Propiedad matemática del ruido gaussiano generado, no del "intento" |

```
                LATENCIA TOTAL DE AUDITORÍA (POR REQUEST)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Guardrail tradicional: 400.00 ms   ███████████████████████████ (100.0%)│
 │ SpecWave LL-Tripwire:    0.09 ms   ▍ (0.02% - >3,000x FASTER)          │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 2. Protocolo Experimental Real

### 2.1 Generación de Datos (IMPORTANTE: 100% sintético, sin texto)

El script **no usa texto, ni jailbreaks, ni prompts reales**. Genera dos nubes de ruido gaussiano con medias opuestas (`tests/benchmark_spectral_safety.py`, líneas 99–116):

```python
# 1. Synthesize Benign Thought Manifolds [N, 64, 768]
benign_embs = torch.randn(num_train_samples // 2, seq_len, d_model, device=device) * 0.5 + 1.2
benign_embs = benign_embs + torch.sin(t) * 1.5

# 2. Synthesize Malicious / Jailbreak Thought Manifolds [N, 64, 768]
malicious_embs = torch.randn(num_train_samples // 2, seq_len, d_model, device=device) * 0.5 - 1.2
malicious_embs = malicious_embs + torch.cos(2 * t) * 1.8
```

- **Clase 0 ("benigno"):** ruido gaussiano con media **+1.2** + seno.
- **Clase 1 ("malicioso"):** ruido gaussiano con media **-1.2** + coseno.

Estas dos nubes son **trivialmente separables**: basta con mirar la media de los valores. Un clasificador lineal llega a 100% en pocos pasos. No hay relación con "intención", "decepción" o "jailbreak".

### 2.2 Arquitectura del Auditor

```python
class SpectralIntentAuditor(nn.Module):
    def __init__(self, half_seq=32, half_dim=384, num_intent_classes=2):
        self.tripwire = nn.Sequential(
            nn.Linear(ll_dim, 128),   # ll_dim = 32*384 = 12,288
            nn.GELU(),
            nn.Linear(128, 2)
        )
```

Un MLP de 2 capas que clasifica el LL subband aplanado. Se entrena 50 pasos con AdamW lr=3e-3.

### 2.3 Evaluación

- 400 muestras de entrenamiento (200 por clase).
- 100 muestras de test "no vistas" (50 por clase), generadas con la **misma distribución** (misma media, mismo ruido).
- Resultado esperado y obtenido: 100% de precisión, porque las clases son linealmente separables por construcción.

---

## ⚡ 3. Desglose de Latencia Medida

Medido en CPU sobre 100 tensores sintéticos:

1. **2D DWT Decomposition Step:** $0.0546\text{ ms}$ por muestra.
2. **LL Subband Tripwire Linear Inference:** $0.0391\text{ ms}$ por muestra.
3. **Total Latency:** **$0.0937\text{ ms}$**.

Esta latencia es real, pero mide el coste de cómputo de un DWT + MLP pequeño sobre tensores de 64×768. No mide "intercepción de intenciones maliciosas".

---

## 💡 4. Interpretación Honesta de los Resultados

1. **El 100% de detección es trivial:** Las dos clases se generan con medias opuestas (+1.2 vs -1.2). Cualquier clasificador (incluso uno que mire solo la media) las separa perfectamente. Esto **no** demuestra capacidad de detectar jailbreaks, engaños o contenido dañino.

2. **La energía del 93.51% en LL es una propiedad del ruido generado:** La distribución de energía entre subbandas de Haar depende de la estructura de los datos de entrada. Con ruido gaussiano + seno/coseno, la mayor parte de la energía cae en LL por construcción. No hay evidencia de que esto se cumpla con embeddings de texto reales.

3. **No hay comparación con guardrails reales:** La tabla compara contra "Llama-Guard / Moderation API" con latencias de 250-600 ms, pero **ninguno de esos sistemas se ejecuta ni se mide**. Son cifras de referencia no verificadas.

4. **Conclusión:** La fase 4 es una demo de que un MLP pequeño puede clasificar dos nubes de ruido separables y que un DWT es rápido. Para que fuera un resultado de safety real haría falta: (a) texto real con jailbreaks reales (p. ej., prompts adversariales conocidos), (b) embeddings de un LLM real, (c) comparación medida contra un guardrail real, y (d) métricas de precisión/recall/F1 sobre datos no triviales.