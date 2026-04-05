# Plan de Ejecución V2 (Post-Cleanup Neural-Only): OpenAI Parameter Golf
## Presupuesto restante: ~$394 USD / RTX 3090 local + RunPod/GCP

---

## 1. CONTEXTO Y REGLAS CLAVE ACTUALIZADAS

### Qué es legal
- **Neural-only optimizations:** Todo lo que pase dentro de la arquitectura del transformer.
- **GPTQ quantization:** LEGAL siempre y cuando la calibración con datos reales ocurra DENTRO de los 600s de training.
- **Complementary training:** Down-weight de tokens fáciles (bigrams).
- **XSA (Cross-Sequence Attention):** En cualquier cantidad de capas.

### Qué es ESTRICTAMENTE ILEGAL (Post-Cleanup)
- **Hashed n-gram caches:** Completamente baneados por no renormalizar correctamente y *leakear* eval tokens (confirmado en issue #140).
- **Cualquier forma de eval leaking:** Mirar hacia adelante al target token para mezclar probabilidades.
- **Two-pass full rescore:** Scorear, armar cache y re-scorear.

### Restricciones duras
- Artefacto ≤ 16,000,000 bytes (código + modelo comprimido).
- Training ≤ 600s en 8xH100 SXM.
- Evaluation ≤ 600s en 8xH100 SXM.
- Deadline: 30 de abril de 2026.

---

## 2. ANÁLISIS DEL LEADERBOARD NEURAL-ONLY

### Frontera Legal Validada (Sin N-gram)
| BPB | Técnica Principal | Referencia |
|---|---|---|
| **1.1086** | Turbo-Muon + EngramLite + mixed int6/int7 GPTQ | PR #1089 |
| 1.1099 | Turbo-Muon + mixed int6/int7 GPTQ | PR #1120 |
| 1.1122 | Coprime-Stride + Full GPTQ + XSA-all | PR #1060 |
| 1.1147 | AR Self-Gen GPTQ + XSA-all + BigramHash | PR #1019 |
| 1.1194 | LeakyReLU² + Legal TTT + Parallel Muon (SOTA viejo) | PR #549 |

### Target realista
**BPB sub-1.108 → Top 3 absoluto.** Teniendo ya una base en ~1.11, el gap es de apenas ~0.002 nats.

---

## 3. STACK TÉCNICO OBJETIVO (V2)

Se elimina por completo el cache n-gram y el TTT. El presupuesto de tiempo de evaluación ahora se invierte 100% en empujar más *steps* de entrenamiento y usar una cuantización más pesada.

```text
┌─────────────────────────────────────────────────────┐
│                    TRAINING (600s)                    │
│                                                       │
│  ① Neural Base Optimizado                            │
│     - 11L GQA Transformer (512d)                      │
│     - LeakyReLU(0.75)² activation                     │
│     - XSA-all: Cross-sequence attention en TODAS      │
│       las 11 capas (reemplaza beneficio de TTT)       │
│     - Embeddings: BigramHash + SmearGate              │
│                                                       │
│  ② Optimizador & Data Loading                         │
│     - Turbo-Muon (con AOL preconditioning)            │
│     - Coprime-Stride data loader (maximiza steps/sec) │
│                                                       │
│  ③ Complementary Training                             │
│     - Bigram stats para down-weight                   │
│                                                       │
│  ④ Full Hessian GPTQ Quantization                     │
│     - Calibración in-training                         │
│     - Full Hessian (reemplaza per-row quant)          │
│     - Compresión zstd → ~15.5MB modelo                │
│                                                       │
├─────────────────────────────────────────────────────┤
│                  EVALUATION (600s)                     │
│                                                       │
│  ⑤ Inferencia Pura (Sin TTT)                          │
│     - Forward pass directo                            │
│     - Aprovecha el time-budget liberado por no hacer  │
│       TTT para correr un modelo ligeramente más       │
│       ancho/profundo si el tamaño lo permite          │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 4. PLAN DE EJECUCIÓN: PIVOT NEURAL

*(Nota: Fases 0 a 4A completadas. Ya tenés el repo, la base neural iterando, y la pipeline en RunPod/GCP operativa).*

### FASE 4B: El Pivot Rápido — RunPod (Día 1-2, ~$14 = 1 hora)

**Objetivo:** Modificar el código actual eliminando código muerto y aplicando los *low-hanging fruits*.

#### Paso 1: Limpieza
- Eliminar toda la lógica de `BackoffNgramMixer`.
- Eliminar la clase `LoRAAdapter` y el *eval loop* de TTT. 
- Volver a un *eval loop* estándar y limpio.

#### Paso 2: Integrar XSA-all y Coprime-Stride
```python
# 1. Cambiar XSA a todas las capas
# Antes: XSA_LAST_N = 4
XSA_LAST_N = 11  # Aplica XSA en las 11 capas del modelo

# 2. Implementar Coprime-Stride Loader
# Intercalar la carga de datos para maximizar la entropía del batch
# y mantener la memoria cacheada en la GPU más eficiente.
def coprime_stride_dataloader(tokens, batch_size, seq_len, stride=137):
    # Implementación basada en PR #1060
    # ...
```
- **Prueba rápida:** Correr 1 seed en RunPod para verificar que XSA en todas las capas no explota la memoria ni el tiempo de evaluación (debería agregar solo ~2ms por step).

### FASE 5: Full Hessian GPTQ — Local/RunPod (Día 3-5, ~$42 = 3 horas)

**Objetivo:** Pasar de una cuantización *per-row* a Full Hessian para recuperar la precisión perdida al dropear TTT.

- Analizar la implementación de GPTQ en los PRs #1060 y #1019.
- La clave: computar la matriz Hessiana inversa durante los últimos 10-15 segundos del *training loop*.
- Asegurar que la calibración se haga *estrictamente* con los batches de entrenamiento dentro de la ventana de 600s.
- **Chequeo de tamaño:** El Full Hessian suele generar matrices ligeramente más densas de comprimir. Validar con `zstd` que el artefacto se mantenga debajo de los 16MB.

### FASE 6: Optimizador Turbo-Muon — RunPod (Día 6-8, ~$70 = 5 horas)

**Objetivo:** Raspar los últimos 0.003 BPB mejorando la convergencia.

- Reemplazar Parallel Muon por Turbo-Muon.
- Integrar *AOL preconditioning* (Alternative Orthogonalization Layout).
- Ajustar el *learning rate schedule* (al tener más *steps* disponibles por haber sacado TTT, podés estirar el *decay*).
- **Hyperparameter tuning:** Dedicar 3-4 horas de RunPod a iterar sobre la tasa de aprendizaje y el ratio de actualización del preconditioner.

### FASE 7: Submission y Pulido Final — RunPod (Día 9-11, ~$42 = 3 horas)

**Objetivo:** Confirmar estabilidad y someter el récord oficial.

- Ejecutar los 3 *seeds* mandatorios (ej. 42, 1337, 7).
- Calcular la media y desviación estándar. Si el promedio rompe los 1.1086 con $p < 0.01$, es récord.
- **Artifact Size Check:** Ejecutar el script oficial para confirmar los bytes exactos.
- Armar el PR detallando cómo XSA-all + Full Hessian elimina la necesidad de TTT.

---

## 5. PRESUPUESTO ACTUALIZADO

| Fase | Estado | Costo Incurrido |
|---|---|---|
| Fases 0-4 (Base + N-gram fallido) | Completada | ~$56 |

| Nueva Fase | Horas RunPod estimadas | Costo USD proyectado |
|---|---|---|
| 4B: Pivot y XSA-all | 1h | $14 |
| 5: Full Hessian GPTQ | 3h | $42 |
| 6: Turbo-Muon | 5h | $70 |
| 7: Submission final | 3h | $42 |
| **Buffer debugging** | 5h | $70 |
| **Total Restante** | **17h** | **~$238** |

**Sobran ~$156 del presupuesto original de $450**, dándote margen más que suficiente para iterar tranquilo y no correr con los costos en GCP/RunPod.

---

## 6. TIPS PRÁCTICOS POST-CLEANUP

- **Más iteraciones es rey:** Sin TTT, el cuello de botella es estrictamente qué tan rápido podés pasar datos por el modelo en 600s. Cualquier optimización de PyTorch (como el `coprime-stride`) que te suba los *tokens/sec* se traduce directamente en mejor BPB.
- **Cuidado con la calibración GPTQ:** Los revisores están con la lupa en esto tras el *cleanup*. Asegurate de que el código de cuantización no toque ni por asomo el dataset de evaluación. Todo el `H_inv` debe computarse en el `train_gpt.py`.
- **Aprovechá la 3090:** XSA-all y el *Coprime-stride* se pueden testear localmente para ver si el *loss* baja más rápido por iteración, antes de gastar saldo en RunPod.