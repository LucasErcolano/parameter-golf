# OpenAI Parameter Golf — Contexto Completo para Continuación

## Quién es Lucas

Lucas Ercolano, estudiante de AI Engineering en UdeSA (Argentina), developer con experiencia en deep learning, CV y agentic AI. Hardware local: RTX 3090 24GB + Xeon. Presupuesto: ~$400 USD restantes en RunPod Spot ($14/hr en 8xH100 SXM). Ya gastó ~$28 en dos corridas de cloud. Background competitivo (olympiad math, game jams, hackathons).

---

## Qué es el competition

OpenAI Parameter Golf: entrenar el mejor LM que quepa en un **artifact de 16MB** (código + modelo comprimido), training ≤ 600s en 8xH100 SXM, eval ≤ 600s separados. Métrica: **bits per byte (BPB)** en FineWeb validation set (~62M tokens). Tokenizer-agnostic. Menor BPB = mejor. Deadline: 30 de abril de 2026.

El artifact = `len(train_gpt.py) + compressed_model_bytes ≤ 16,000,000 bytes`.

---

## Reglas de legalidad (crítico)

### LEGAL
- **Score-first TTT:** Fine-tune LoRA en val tokens YA evaluados. Regla de issue #402.
- **Causal n-gram cache:** Construir cache solo con tokens ya scorados. Update DESPUÉS de scoring. Issue #677 "directionally legal".
- **N-gram tables del training set empaquetadas en artifact:** Legal, pagás esos bits.
- **GPTQ quantization:** Legal si calibración ocurre DENTRO de los 600s de training.
- **Complementary training:** Down-weight tokens predecibles por bigrams en loss. Legal.
- **Cualquier stride / sliding window en eval:** Legal.
- **Shared n-gram tables entre GPUs (X-WING):** Legal.

### ILEGAL (descalificación)
- **Two-pass full rescore:** Scorear todo, construir cache completo, re-scorear. ILEGAL.
- **Pre-eval TTT:** Entrenar en val tokens ANTES de evaluarlos. ILEGAL.
- **GPTQ calibration en eval time:** Usar training data durante eval. ILEGAL.
- **Hindsight selection:** Usar true next token para decidir alpha de blending. ILEGAL.
- **Paid prefix:** Comprimir val data en artifact. ILEGAL.

### Records
- Nuevo SOTA: superar actual por ≥0.005 nats a p < 0.01 (3 seeds).
- El #1 del leaderboard (Nacrith, BPB 0.0000) fue cerrado por eval leaking ilegal.
- SOTA oficial neural puro: PR #549, 1.1194 BPB.

---

## Stack técnico implementado

El código sigue la línea **PR #549 → #803 → #1033**. `train_gpt_phase3.py` es un fork de #1033 parcheado para legalidad, quant y runtime. NO hereda del `train_gpt.py` del repo base pero tiene todos los componentes del stack #549.

### Componentes implementados y funcionando

**Modelo neural (PR #549 stack):**
- 11L GQA Transformer, 512d, 8 heads, 4 KV heads
- MLP 3.5x con LeakyReLU(0.75)² (actualmente, debe bajar a 3.0x — ver bloqueantes)
- Parallel Muon optimizer
- SmearGate + BigramHash(2048) + OrthoInit
- XSA en últimas 4 layers
- Value-Residual Embeddings (VE128) en layers 9,10
- Partial RoPE (16/64 dims), LayerNorm Scale
- EMA (0.997) + SWA durante warmdown
- GPTQ-lite int6 con clip search (5 percentiles por row)
- LZMA compression
- 29,877,350 params con MLP 3.5x (~27M con MLP 3.0x)

**Complementary Training (PR #803):**
- Bigram stats → loss reweighting: `w_i = 1 - alpha * p_bigram(token_i)`
- Modelo se especializa en lo que n-gram NO puede predecir

**Causal BackoffNgramMixer (PR #803/#1033):**
- Orders 2 a max_order, hash buckets
- Entropy-adaptive alpha blending
- Score-first: evaluar → blendear → BPB → LUEGO update cache
- Backoff: intenta orden más alto primero, cascade down

**LoRA TTT (PR #461/#549/#1033):**
- LoRA en Q/K projections
- Score-first: solo entrena en tokens ya evaluados
- Actualmente DESHABILITADO por default (TTT_ENABLED=0) — no aporta localmente y agrega overhead

**Eval pipeline:**
- Sliding window con stride configurable (default 128, probado hasta 512)
- Batched forward passes (EVAL_BATCH_SEQS=32)
- torch.compile con fallback a eager
- torch.inference_mode() en scoring
- Multi-GPU con DDP guards
- Checkpointing para RunPod Spot (save cada 60s a /workspace)

---

## Archivos clave

```
parameter-golf-main/
├── train_gpt_phase3.py          # Código principal de desarrollo (authoritative)
├── train_gpt.py                 # Baseline original del repo (intacto, no se usa)
├── RUNPOD_PLAYBOOK.md           # Comandos exactos para cloud
├── REVIEW_PROMPT.md             # Prompt para auditoría por otro agente
├── SUMMARY.md                   # Resumen de corridas cloud
├── data/                        # Datos FineWeb (1 shard smoke local)
├── logs/
│   ├── baseline3_smoke.txt
│   ├── phase2_smoke.txt
│   ├── phase3_ttt_only.txt
│   ├── mlp35_artifact_bench_262k_2026-04-03.txt
│   ├── mlp35_smoke_200_fit.txt
│   ├── runtime_stride_bench_2026-04-03.txt
│   ├── runtime_ttt_bench_2026-04-03.txt
│   ├── cloud_seed42_train.txt       # Primera corrida cloud
│   ├── cloud_seed42_evalonly_fix.txt # Primera corrida cloud eval
│   ├── cloud_seed42_train2.txt      # Segunda corrida cloud
│   └── cloud_seed42_eval512.txt     # Segunda corrida cloud eval
└── records/track_10min_16mb/2026-04-03_PRAligned_ComplementaryBackoff_LoRATTT_3090/
    ├── train_gpt.py             # Submission package (sync con phase3)
    ├── README.md
    ├── LOCAL_DIAGNOSTICS.md
    ├── submission.json
    ├── eval/
    │   └── eval.sh
    └── logs/
```

---

## Historial de resultados

### Smoke local (RTX 3090, 200 iters, datos parciales)

| Test | BPB | Nota |
|---|---|---|
| Baseline smoke 65K tokens | 6.2638 | Solo baseline del repo |
| Phase 1 pre-quant 65K | 5.9172 | +LeakyReLU +complementary |
| Phase 1 quant roundtrip 65K | 4.0995 | Delta quant: +0.0009 ✓ |
| Phase 2 n-gram 32K | 5.5159 | Score-first mixer funciona |
| Phase 2 n-gram 65K | 3.6884 | N-gram escala con tokens |
| Phase 2 n-gram 262K | 3.2904 | Sigue bajando |
| Phase 2 n-gram 1M | 3.0590 | Sigue bajando |
| MLP 3.5x n-gram 262K | 3.0024 | Mejora marginal vs 3.0x |
| TTT sin n-gram 32K | 6.5050 | Aporta marginal solo |
| TTT + n-gram 32K | 5.5159 | Sin efecto extra en smoke |

### Cloud real (8xH100 SXM, seed=42, training completo 600s)

**Primera corrida:**
- Training: step 5117, diag_bpb=1.1429, q_rt_bpb=1.1511, q_sw_bpb=1.1276
- Eval ngram PARCIAL (cortó a 580s): 0.4982 BPB
- Bug encontrado y corregido: torch.compile recompilaba por batch sizes dinámicos

**Segunda corrida:**
- Training: step 4750, diag_bpb=1.1465, q_rt_bpb=1.1548, q_sw_bpb=1.1314
- Eval stride=256 PARCIAL (cortó a 580s): 0.5021 BPB
- Eval stride=512 PARCIAL (cortó a 590s): 0.4948 BPB
- **Artifact: 19,164,311 bytes → EXCEDE 16MB**

### Verificaciones pasadas
- Determinismo: exact match en 5,000 tokens ✓
- Causalidad: BPB de prefijo no cambia al mutar sufijo futuro ✓
- Quant roundtrip: delta +0.0009 BPB ✓

---

## DOS BLOQUEANTES ACTUALES (lo que hay que resolver)

### BLOQUEANTE 1: Artifact > 16MB

**Diagnóstico:** 29.8M params (MLP 3.5x) × 6 bits = 22.4MB raw. LZMA comprime a 85% → 19MB. Un modelo entrenado de verdad tiene weights de alta entropía que no comprimen bien (el smoke local de 5MB era con weights casi-inicializados).

**Plan acordado:**
1. Volver a MLP 3.0x (~27M params → ~20.25MB raw int6)
2. Cuantizar MLP weights a int5, attention weights a int6 (mixed precision)
   - Attention ~9.5M params × 6/8 = 7.1MB
   - MLP ~16M params × 5/8 = 10.0MB
   - Otros ~1.5M × 6/8 = 1.1MB
   - Total raw: ~18.2MB, con 85% compresión: ~15.5MB + 120KB código = ~15.6MB ✓
3. Probar zstd (level 22) vs lzma y elegir el menor
4. Verificar con modelo entrenado 2000+ iters que comprimido < 15MB

### BLOQUEANTE 2: Eval no entra en 600s

**Diagnóstico:** stride 256→512 solo ganó 10s (580→590s). El forward neural ya no es el bottleneck. El cuello de botella es el **loop Python per-token del n-gram mixer** procesando ~7.75M tokens por GPU. ~14K tokens/segundo = velocidad de Python puro.

**Plan acordado:**
1. Vectorizar n-gram predict con numpy (batch predict para chunks enteros)
2. Vectorizar n-gram update con numpy (np.add.at para batch updates)
3. Vectorizar BPB computation
4. Considerar numba @njit para hash computation si numpy no alcanza
5. Target: 262K tokens en < 5s en 3090 (extrapola a ~400s en 8xH100)

---

## PRs DE REFERENCIA (ya leídos/estudiados)

**Stack base:**
- #198 (XSA), #287 (RoPE+LN), #315 (frontier base), #374 (GPTQ-lite+EMA), #414 (standard frontier), #493 (LeakyReLU ablation), **#549 (SOTA neural, código de referencia principal)**

**N-gram:**
- #659 (cerrado, ilegal — qué NO hacer), #702 (primer legal), #706 (5-gram simple), #715 (entropy refinement), **#727 (primer sub-1.0)**, #741 (+TTT), **#779 (BackoffNgramMixer reference impl)**, **#798 (entropy gating per-order)**, **#800 (X-WING shared tables)**

**TTT:**
- #267 (primer causal), **#461 (SGD+momentum framework)**, #473 (legal validation), **#548 (batched LoRA)**, #549 (integrado), #779 (drift-free)

**Complementary training:**
- **#803 (complementary + backoff, LA PR clave)**, **#1033 (+ TTT, base del fork de Lucas)**

**Research:**
- **#831 (why architectures fail at 16MB)** — throughput-quantization co-optimization constraint
- Issue #140 (live AI commentary — biblia de la competencia)
- Issue #402 (TTT legality ruling)
- Issue #677 (enforcement sweep)

---

## DECISIONES YA TOMADAS

1. **Primera corrida cloud sin TTT** (TTT_ENABLED=0) — correcto, TTT no aporta localmente y agrega ~160s
2. **Stride 128-256** en eval — stride 128/256/512 dan mismo BPB gracias al mixer
3. **No implementar packed n-gram tables del training set** por ahora — el cache causal solo funciona bien
4. **No implementar X-WING** (shared n-gram tables entre GPUs) todavía — simplificar primero
5. **RunPod Spot** ($14/hr) en vez de on-demand ($21.52/hr) — checkpointing implementado
6. **BIGRAM_VOCAB_SIZE=2048** (no 1536 como #549) — no vale la pena cambiar

---

## PLAN DE FASES RESTANTES (del plan original)

- **Fase 4 (primer cloud sprint):** EN CURSO — necesita resolver los 2 bloqueantes antes de la tercera corrida
- **Fase 5 (iteración hyperparams):** Pendiente — iterar n-gram params, complementary training, TTT en cloud
- **Fase 6 (técnicas avanzadas):** Pendiente — X-WING shared tables, Cubric per-order alpha, three-tier token weighting
- **Fase 7 (submission final):** Pendiente — 3 seeds, PR

---

## CHECKLIST ANTES DE TERCERA CORRIDA EN CLOUD

```
ARTIFACT (< 16MB):
[ ] MLP 3.0x (volver de 3.5)
[ ] Int5 para MLP weights, int6 para attention (mixed precision quantization)
[ ] Probar zstd vs lzma, elegir el mejor
[ ] Smoke local: entrenar 2000+ iters, medir artifact comprimido
[ ] Verificar: artifact < 15MB (dejar 1MB de margen)

EVAL TIME (< 600s):
[ ] N-gram predict vectorizado con numpy (batch predict chunks enteros)
[ ] N-gram update vectorizado con numpy (np.add.at)
[ ] BPB computation vectorizado
[ ] Opcional: numba @njit para hash computation
[ ] Benchmark: 262K tokens < 5s en 3090
[ ] Extrapolado: < 400s en 8xH100

INTEGRACIÓN:
[ ] Pipeline end-to-end local con ambos fixes
[ ] eval.sh actualizado
[ ] Submission train_gpt.py sincronizado
[ ] Correr 3 seeds si todo pasa
```

---

## BUDGET RESTANTE

~$400 USD en RunPod Spot. A $14/hr = ~28.5 horas de 8xH100. Ya gastó ~$28 en 2 corridas (~2 horas). Cada run (train + eval) tarda ~25 min, así que tiene ~68 runs restantes. Abundante para iterar.

---

## QUÉ PEDIRLE AL NUEVO CHAT

"Estoy trabajando en una submission para OpenAI Parameter Golf. Adjunto el contexto completo del proyecto en MIGRATION_CONTEXT.md. Mis archivos principales son train_gpt_phase3.py (código) y los logs en la carpeta logs/. Tengo dos bloqueantes que resolver antes de la siguiente corrida en cloud: (1) el artifact comprimido mide 19MB y debe ser < 16MB, y (2) la eval de 62M tokens no entra en 600s porque el n-gram mixer es un loop Python lento. Necesito implementar mixed precision quantization (int5 MLP + int6 attention) y vectorizar el n-gram mixer con numpy. Adjunto los archivos."
