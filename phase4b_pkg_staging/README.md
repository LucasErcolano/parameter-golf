# PR-Aligned Phases 1-3 Local Record

Local packaging of the PR-aligned `train_gpt.py` stack up to phase 3.

Stack:
- Phase 1 base aligned to the `#549/#803/#1033` lineage:
  - 11L / 512d / GQA 8/4
  - MLP 3.5x
  - XSA-4
  - Partial RoPE 16
  - LN scale
  - BigramHash(2048)
  - VE128
  - VRL
  - SmearGate
  - LeakyReLU^2
  - EMA + SWA
- mixed int6 + lzma artifact export
- packed int6 payloads inside the mixed-quant artifact to avoid storing int6 weights as raw int8 bytes
- Phase 2:
  - backward-looking BackoffNgramMixer
  - 4M buckets default
  - entropy-adaptive alpha `0.20 + 0.55 * sigmoid(2 * (H - 3.0))`
- Phase 3:
  - score-first LoRA TTT on Q/K
  - per-document reset using BOS boundaries
  - no token-stream TTT
  - no min-NLL / no multi-epoch hindsight selection

Local compatibility changes:
- `COMPILE_MODEL=0` by default on Windows
- `COMPILE_MUON=0` by default on Windows
- `ADAM_FUSED=0` by default on Windows
- `torch.compile` auto-falls back to eager when Triton is unavailable
- data/tokenizer defaults now resolve from `__file__`, so the package works from `records/...`
- TTT DDP adaptation now truncates per-chunk sequence counts to a multiple of `WORLD_SIZE` so all ranks execute the same number of `all_reduce` steps
- `eval_only` no longer instantiates the training loader or reads training shards
- final checkpoint application now prefers SWA when available, otherwise EMA
- eval passes respect `EVAL_TIMEOUT_SECONDS` and can return a partial score instead of hanging to SIGKILL
- training supports Spot-safe checkpoint/resume through `CKPT_DIR`, `CKPT_EVERY_SECS`, and `RESUME_CKPT`
- final artifacts are copied to `CKPT_DIR/artifact_seed<SEED>/` when checkpointing is enabled
- eval defaults now target `EVAL_STRIDE=256`
- sliding and ngram eval pad the last batch to a fixed `batch_seqs`, avoiding `torch.compile` recompile-limit failures on H100

Local diagnostics run on RTX 3090:
- Package size:
  - `train_gpt.py`: `102,005` bytes
  - `final_model.int6.ptz`: `5,006,588` bytes
  - total: `5,108,593` bytes
- Current default model:
  - `29,877,350` params
  - `11L / 512d / MLP 3.5 / heads 8 / kv 4`
- Quant roundtrip on 65,536 val tokens:
  - previous `MLP 3.0` reference: `diag_bpb=4.0986`, `q_rt_bpb=4.0995`, delta `+0.0009`
- `MLP 3.5` local smoke with reduced training batch (`TRAIN_BATCH_TOKENS=131072`, wallclock-capped at `600s`):
  - stopped at step `142/200` by wallclock cap
  - `diag_bpb=3.4483`
  - `q_rt_bpb=3.4579`
  - quant delta `+0.0096`
  - `ngram_bpb=3.2847` on `65,536` tokens
- Artifact payload roundtrip through `lzma`:
  - exact payload equality passed
- N-gram scaling with quantized artifact:
  - `65,536 -> 3.6884 BPB`
  - `262,144 -> 3.2904 BPB`
  - `1,048,576 -> 3.0590 BPB`
- Fresh-process benchmark with current `MLP 3.5` artifact:
  - `262,144` tokens, `stride=128`: `3.0024 BPB` in `59.51s`
  - extrapolated `8xH100 ~= 503s`
- N-gram + LoRA TTT at `262,144` tokens:
  - `3.2904 BPB`
- Determinism:
  - exact-match BPB passed on 5,000 tokens
- Causality:
  - prefix invariance passed on 4,096-token prefix with mutated suffix
- Local profile on 65,536 tokens:
  - neural forward: `32.11s`
  - n-gram score/blend: `0.51s`
  - n-gram update: `0.09s`
  - LoRA update over one `32K` chunk, `4` epochs: `5.84s`
- Extrapolation on single RTX 3090:
  - n-gram-only `1,048,576` tokens: `566.8s`
  - n-gram-only `62M` tokens: not locally viable
  - LoRA-TTT over `62M` tokens at current implementation: not locally viable

Runtime optimization pass:
- default eval stack now uses `EVAL_STRIDE=128`
- score batch and TTT-train batch are separated
- sliding eval stages contiguous spans on GPU instead of per-window host copies
- post-patch `262,144`-token `ngram-only` timings:
  - `stride=64`: `130.48s` (`~1102s` extrapolated on `8xH100`)
  - `stride=128`: `67.71s` (`~572s` extrapolated on `8xH100`)
  - `stride=256`: `33.78s` (`~285s` extrapolated on `8xH100`)
  - `stride=512`: `17.11s` (`~145s` extrapolated on `8xH100`)
- post-patch `262,144`-token `ngram + LoRA TTT` at `stride=128`:
  - `86.39s` (`~730s` extrapolated on `8xH100`)
  - no local BPB gain over n-gram-only
- Package smoke from `records/...`:
  - direct `EVAL_ONLY=1 python train_gpt.py` passed
  - tokenizer/data path resolution from package dir passed
  - `eval:ngram bpb:3.4191` on `4,096` tokens
- WSL validation:
  - `torchrun --standalone --nproc_per_node=1 train_gpt_phase3.py` passed on Ubuntu WSL
  - `bash eval/eval.sh` with `NPROC_PER_NODE=1` passed from the packaged `records/...` directory
  - under WSL this host exposes `torch 2.11.0+cu128`, CUDA, and Triton correctly
  - `EVAL_ONLY=1` with `TRAIN_FILES` pointed at a missing path still passed, confirming eval no longer touches training shards

Known local limitation:
- Full `62M` token eval with `stride=64` is too slow on a single RTX 3090.
- The first cloud run should prefer `ngram-only` over `LoRA TTT` unless a longer training run shows a clear TTT gain.
- The Windows launcher on this host is still flaky, but the equivalent checks passed under WSL, which is the relevant Linux-like execution path for cloud parity.
