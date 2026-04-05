# Phase 3 Local Pipeline

Local submission-style snapshot built from `train_gpt_phase3.py`.

## What is included

- `train_gpt.py`: single-file training/eval pipeline with:
  - LeakyReLU(0.75)^2
  - complementary training loss reweighting
  - EMA before export
  - mixed `int6/int8` GPTQ-lite export with exact save/load validation
  - score-first causal n-gram mixer
  - score-first LoRA TTT
  - utility run modes for `eval_only`, `quant_diag`, `causality`, `determinism`, `profile`, and `eval_curve`
- `eval/eval.sh`: artifact evaluation entrypoint
- `requirements.txt`
- `logs/`: local reproducibility logs

## Local smoke result

Smoke training run:

- `SEED=1337`
- `ITERATIONS=200`
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_MAX_TOKENS=65536`
- `MLP_LEAKY_SLOPE=0.75`
- `COMPLEMENT_ALPHA=0.5`
- `QUANT_BITS=6`
- `NGRAM_ENABLED=1`
- `TTT_ENABLED=1`

Observed metrics on the produced artifact:

- Post-EMA neural: `val_bpb=2.83845947`
- Post-quant artifact neural: `val_bpb=2.83641713`
- Post-quant + n-gram: `val_bpb=2.62732295`
- Post-quant + n-gram + TTT: `val_bpb=2.58189927`
- Artifact size: `8,091,088` bytes

## Extra local checks

- Causality test passed for `1000` tokens
- Determinism test passed for `5000` tokens
- Long-curve results on the same artifact:
  - neural `65,536 -> 2.83641713`
  - neural `262,144 -> 2.81492155`
  - neural `1,048,576 -> 2.86609464`
  - neural `4,194,304 -> 2.88199222`
  - n-gram `65,536 -> 2.62732295`
  - n-gram `262,144 -> 2.68575568`
  - n-gram `1,048,576 -> 2.81575999`
  - n-gram `4,194,304 -> 2.84555694`
  - TTT every 8 chunks `1,048,576 -> 2.74791652`
  - TTT every 8 chunks `4,194,304 -> 2.74205832`

## Notes

- The current training path is DDP-ready.
- Score-first n-gram and TTT evaluation are still local-process oriented; `TTT_EVERY_N_CHUNKS` was added to thin adaptation frequency during long evals.
- Local profile with `TTT_EVERY_N_CHUNKS=8` extrapolated roughly:
  - neural-only component `~848s / 62M` on a single 3090-equivalent process
  - full score-first loop `~2206s / 62M` on a single 3090-equivalent process
  - TTT updates `~1202s / 62M` at one adaptation every 8 chunks
- The logs in `logs/` are the authoritative record of what was run locally.
