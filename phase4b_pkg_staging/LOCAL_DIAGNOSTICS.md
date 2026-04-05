# Local Diagnostics Summary

## Quantization

- Artifact bytes: `5,006,588`
- Code bytes: `102,005`
- Total bytes: `5,108,593`
- Payload roundtrip through `lzma`: exact equality passed
- `diag_bpb` on `65,536` tokens: `4.0986`
- `q_rt_bpb` on `65,536` tokens: `4.0995`
- Quantization delta: `+0.0009 BPB`

## Current Model

- Default architecture:
  - `11L / 512d / MLP 3.5 / heads 8 / kv 4`
- Parameter count:
  - `29,877,350`
- Size estimate:
  - `59.75 MB` in fp16
  - `22.41 MB` raw int6 before compression

## Eval Curve

- N-gram only:
  - `65,536` tokens: `3.6884 BPB` in `32.76s`
  - `262,144` tokens: `3.2904 BPB` in `134.34s` before runtime patches
  - `1,048,576` tokens: `3.0590 BPB` in `566.78s`
- N-gram + LoRA TTT:
  - `262,144` tokens: `3.2904 BPB` in `138.38s` before runtime patches

## Legality Checks

- Determinism on `5,000` tokens:
  - exact match passed
  - `4.126029123941933 BPB`
- Prefix causality test:
  - prefix length `4,096`
  - mutated suffix leaves prefix BPB unchanged
  - exact match passed

## Runtime Profile

- Profiled on `65,536` tokens:
  - neural forward: `32.11s`
  - n-gram score/blend: `0.51s`
  - n-gram update: `0.09s`
- LoRA update cost:
  - one `32K` chunk, `4` epochs: `5.84s`

## Runtime Optimization Pass

- Code changes:
  - default `EVAL_STRIDE=128`
  - separate `EVAL_BATCH_SEQS=32` from `TTT_TRAIN_BATCH_SEQS=8`
  - contiguous batch staging on GPU for sliding eval and n-gram/TTT eval
  - `ngram-only` path no longer pays LoRA forward overhead
  - `torch.compile` now degrades cleanly to eager when Triton is unavailable
- `262,144`-token `ngram-only` benchmark after patches:
  - `stride=64`: `3.3085 BPB` in `130.48s`, extrapolated `8xH100 ~= 1102s`
  - `stride=128`: `3.3054 BPB` in `67.71s`, extrapolated `8xH100 ~= 572s`
  - `stride=256`: `3.3055 BPB` in `33.78s`, extrapolated `8xH100 ~= 285s`
  - `stride=512`: `3.3052 BPB` in `17.11s`, extrapolated `8xH100 ~= 145s`
- `262,144`-token `ngram + LoRA TTT` after patches:
  - `stride=128`: `3.3054 BPB` in `86.39s`, extrapolated `8xH100 ~= 730s`
- Peak VRAM:
  - `ngram-only`: `~10.97 GB`
  - `ngram + TTT`: `~17.94 GB`

## MLP 3.5 Smoke

- Local training smoke on one RTX 3090 needed reduced `TRAIN_BATCH_TOKENS=131072`.
- Run ended at step `142/200` because of the `600s` wallclock cap.
- Final metrics on `65,536` validation tokens:
  - `diag_bpb=3.4483`
  - `q_rt_bpb=3.4579`
  - `q_sw_bpb=3.4659` at `stride=128`
  - `ngram_bpb=3.2847`
- Artifact size after the smoke:
  - `5,006,588` bytes
- Fresh-process eval benchmark on the current artifact:
  - `262,144` tokens, `stride=128`: `3.0024 BPB` in `59.51s`
  - extrapolated `8xH100 ~= 503s`

## Package Validation

- `train_gpt.py` inside `records/...` now resolves `data/` and tokenizer paths by walking up from `__file__`.
- TTT adaptation uses an even distributed sequence span helper, truncating chunk-local TTT seq counts to a multiple of `WORLD_SIZE` to prevent mismatched `all_reduce` counts across ranks.
- `eval_only` no longer creates `DistributedTokenLoader`; eval can run with `TRAIN_FILES` set to a missing path.
- Final model selection now applies SWA when warmdown averaging exists, otherwise falls back to EMA.
- Eval timeout guard is controlled by `EVAL_TIMEOUT_SECONDS` (default `580`).
- Spot support:
  - periodic checkpoints via `CKPT_DIR` and `CKPT_EVERY_SECS`
  - resume restores model, optimizer, EMA/SWA, RNG, training loader position, and complementary-training tracker state
  - final artifacts are copied to `CKPT_DIR/artifact_seed<SEED>/`
- Cloud fixes after the first 8xH100 run:
  - int6 weights are now bit-packed before `lzma`, instead of being stored as one-byte `int8` values
  - default `EVAL_STRIDE` moved to `256`
  - sliding and ngram eval use fixed-size padded batches to avoid `torch.compile` recompile-limit failures
- Direct package smoke passed:
  - `EVAL_ONLY=1 python train_gpt.py`
  - `4,096` tokens
  - `eval:ngram bpb:3.4191`
- WSL validation passed:
  - `torchrun --standalone --nproc_per_node=1 train_gpt_phase3.py`
  - `3` train steps, quant export, `q_rt`, `q_sw`, `q_s64`
  - log: `logs/wsl_torchrun_dryrun.txt`
  - `bash eval/eval.sh` from the packaged `records/...` directory with `NPROC_PER_NODE=1`
  - log: `logs/wsl_record_evalsh.txt`
  - `bash eval/eval.sh` from the packaged directory with `TRAIN_FILES=/definitely/missing/...`
  - log: `logs/wsl_eval_no_train_access_serial.txt`
  - timeout smoke with partial return:
  - log: `logs/wsl_eval_timeout_smoke.txt`
- The old launcher failure is specific to the Windows-side `torchrun` environment on this host, not to the script or the package.

## Main Conclusion

- Quantization is no longer the blocker.
- `ngram-only` with `stride=128` is now within the rough `8xH100 / 600s` eval budget.
- `LoRA TTT` is still runtime-negative locally: no BPB gain at `262K` tokens and it pushes the extrapolated eval past budget.
- The first cloud run should use the runtime-safe n-gram path first, then re-test TTT only if the stronger training run creates measurable gain.
