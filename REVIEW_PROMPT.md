# Code Review Prompt: OpenAI Parameter Golf Submission

You are reviewing a submission for the **OpenAI Parameter Golf** competition. The author has built a complete training + evaluation pipeline and needs you to audit it for correctness, legality, and readiness before deploying to cloud (8xH100 RunPod, $14/hr). **Read every file thoroughly before responding.**

---

## Competition Rules (non-negotiable)

OpenAI Parameter Golf challenges participants to train the best language model that fits in a **16MB artifact** (code + compressed model combined, cap is 16,000,000 bytes), training in under **10 minutes (600s) on 8xH100 SXM**, evaluated by compression on the FineWeb validation set measured in **bits per byte (BPB)** — lower is better, tokenizer-agnostic. Evaluation also has a separate 10-minute budget.

### What is LEGAL
- **Score-first TTT (test-time training):** Fine-tune (e.g. LoRA) on validation tokens that have ALREADY been evaluated. The model must score a token BEFORE it can use that token for adaptation. Ruled legal per issue #402 by @0hq.
- **Causal n-gram cache:** Build an n-gram frequency cache during evaluation using ONLY tokens that have already been scored. Update the cache AFTER scoring each token. Ruled "directionally legal" per issue #677 by @valerio-oai.
- **N-gram tables from training data packed into the artifact:** Legal — you pay for those bits within the 16MB limit.
- **GPTQ quantization:** Legal IF calibration happens WITHIN the 600s training budget, using training data only.
- **Complementary training:** Down-weighting tokens predictable by bigram statistics during training loss computation. Legal.
- **Any sequence length / sliding window / stride in eval:** Legal. No restriction on eval method.
- **Shared n-gram tables across GPUs:** Legal.
- **EMA / SWA weight averaging:** Legal.

### What is ILLEGAL (causes disqualification)
- **Two-pass full rescore:** Scoring all tokens, building a cache from ALL tokens, then re-scoring everything. This leaks future eval tokens. ILLEGAL.
- **Pre-eval TTT:** Training on validation tokens BEFORE evaluating them. ILLEGAL.
- **GPTQ calibration at eval time:** Using training data during the 600s eval budget. ILLEGAL.
- **Hindsight selection:** Comparing n-gram prediction vs neural prediction against the TRUE next token to decide blending alpha. This uses the ground truth answer to choose the method, which leaks. ILLEGAL.
- **Paid prefix:** Compressing validation data into the 16MB artifact. ILLEGAL.
- **Accessing training data during evaluation** unless those bits are paid for within the 16MB artifact. ILLEGAL.
- **Network calls during evaluation.** ILLEGAL.

### Record requirements
- New SOTA must beat current best by ≥0.005 nats at p < 0.01 significance (typically 3 seeds).
- Artifact = code bytes (`train_gpt.py`) + compressed model bytes ≤ 16,000,000.
- Training ≤ 600s on 8xH100 SXM.
- Evaluation ≤ 600s on 8xH100 SXM (separate budget).
- Must be reproducible from the submitted code.

---

## What the author built (target stack)

The submission follows the lineage **PR #549 → #803 → #1033** from the competition:

### Architecture (PR #549 SOTA neural stack)
- 11-layer GQA Transformer, 512d, 8 heads, 4 KV heads
- MLP 3.5x with LeakyReLU(0.75)² activation
- Parallel Muon optimizer (not standard Adam)
- SmearGate + BigramHash(2048) + OrthoInit
- XSA (Cross-Sequence Attention) on last 4 layers
- Value-Residual Embeddings (VE128) on layers 9,10
- Partial RoPE (16/64 dims)
- LayerNorm Scale
- EMA (decay=0.997) + SWA during warmdown
- GPTQ-lite int6 quantization with clip search, calibrated within training budget
- LZMA compression of quantized weights

### Complementary Training (PR #803)
- During training, compute bigram predictability for each token
- Down-weight easy tokens in the loss: `w_i = 1 - alpha * p_bigram(token_i)`
- This makes the neural model specialize on tokens the n-gram cache CAN'T predict

### Causal BackoffNgramMixer (PR #803 / #1033)
- Multi-order n-gram cache (orders 2 through max_order) with hash buckets
- Entropy-adaptive alpha blending: when the neural model is uncertain (high entropy), trust n-grams more
- **Score-first protocol:** for each token, (1) score with neural model, (2) blend with n-gram prediction from ALREADY-SEEN tokens, (3) compute BPB, (4) THEN update the n-gram cache with the true token
- Per-order backoff: try highest order first, cascade down on miss

### Score-First LoRA TTT (PR #461 / #549 / #1033)
- LoRA adapters on Q/K projections
- After scoring a chunk of tokens, fine-tune LoRA on those already-scored tokens
- Currently disabled by default (TTT_ENABLED=0) because it doesn't help locally and adds ~160s eval overhead
- When enabled, uses SGD with momentum, multiple epochs per chunk

### Eval pipeline
- Sliding window with configurable stride (default 128)
- Batched forward passes (EVAL_BATCH_SEQS=32)
- torch.compile with best-effort fallback to eager mode
- torch.inference_mode() during scoring
- Multi-GPU ready with DDP guards

---

## Files to review

The main files are:

1. **`train_gpt_phase3.py`** — The full training + evaluation script (development version). This is the authoritative implementation.

2. **`records/track_10min_16mb/2026-04-03_PRAligned_ComplementaryBackoff_LoRATTT_3090/train_gpt.py`** — The submission-packaged version. Should be functionally identical to train_gpt_phase3.py.

3. **`records/.../eval/eval.sh`** — The evaluation shell script.

4. **`records/.../README.md`** — Submission writeup.

5. **`RUNPOD_PLAYBOOK.md`** — Cloud execution playbook.

6. **Log files in `logs/`** — Runtime benchmarks and smoke test results.

---

## What to verify (your review checklist)

### A. LEGALITY (most critical — disqualification risk)

1. **N-gram cache causality:** Verify that the n-gram cache is ONLY updated with tokens AFTER they have been scored. The flow must be: predict → blend → compute BPB → THEN update cache. Search for the eval loop and trace the order of operations. Flag ANY place where a token could leak into the cache before being scored.

2. **TTT legality:** If TTT is implemented, verify that it only trains on tokens that have already been evaluated. The model must score under `torch.inference_mode()` or `torch.no_grad()` first, then train on those scored tokens afterward. Flag any place where TTT could access unseen validation tokens.

3. **No training data access during eval:** Verify that during the evaluation phase, no training data is loaded, read, or used. GPTQ calibration must happen during training, not eval. The only data accessed during eval should be the validation set and the saved artifact.

4. **No hindsight selection:** The alpha blending between neural and n-gram predictions must NOT use the true next token to decide the blend weight. Alpha should be computed from the neural model's entropy or other features that don't involve the ground truth.

5. **No two-pass rescoring:** Verify there is no mechanism that scores all tokens, builds a complete cache, and then re-scores. The eval must be single-pass (or at most, score-first where TTT adapts on already-scored tokens for future chunks).

### B. CORRECTNESS

6. **Quantization roundtrip:** The save/load pipeline for int6 quantized weights must be lossless (compress → decompress → dequantize should give the same weights). Check the serialization code. The author reports delta of +0.0009 BPB which is acceptable.

7. **BPB computation:** Verify the bits-per-byte calculation is correct. BPB = total_bits / total_bytes, where bits come from -log2(probability_of_true_token) and bytes come from UTF-8 byte length of detokenized tokens. This must be tokenizer-agnostic.

8. **N-gram hash function:** Check for hash collision handling. With 4M buckets and up to 62M token updates, collisions are expected. Verify that collisions degrade gracefully (overwrite/accumulate) rather than causing incorrect predictions.

9. **Complementary training:** Verify that bigram statistics are computed from training data (not validation), and that the loss reweighting formula is correct: tokens with high bigram predictability get lower weight.

10. **LoRA implementation:** If present, verify that LoRA adapters are correctly applied (low-rank A×B decomposition), properly initialized, and that the base model weights are NOT modified during TTT (only LoRA weights change).

11. **EMA/SWA:** Verify that EMA is maintained during training and that the quantized/saved model uses EMA weights (not raw training weights). SWA should average checkpoints during warmdown.

12. **Sliding window eval:** Verify that stride is correctly applied — the model evaluates tokens at positions stride apart, and the BPB is computed only for the last `stride` tokens of each window (not the entire window).

### C. MULTI-GPU READINESS

13. **DDP compatibility:** The code must work with `torchrun --standalone --nproc_per_node=8`. Check for:
    - `dist.init_process_group()` call (with graceful fallback for single-GPU)
    - Rank-aware logging (only rank 0 prints)
    - Rank-aware artifact saving (only rank 0 saves)
    - Proper device assignment (`cuda:{local_rank}`)

14. **Eval distribution:** In multi-GPU eval, each GPU should process a SEPARATE slice of the validation set (not the same data). The final BPB should be aggregated correctly via all_reduce (sum of bits / sum of bytes, not average of per-GPU BPBs).

15. **N-gram cache in multi-GPU:** Each GPU should have its own cache for its slice of data (simple mode), OR caches should be synchronized via all_reduce (X-WING mode). Either is fine, but the implementation must be consistent.

### D. ARTIFACT SIZE AND SUBMISSION FORMAT

16. **Artifact size:** `len(open('train_gpt.py','rb').read()) + compressed_model_size < 16,000,000`. The author reports 5,108,593 bytes total — well under limit. Verify the computation is correct.

17. **Submission structure:** The records folder should contain:
    - `train_gpt.py` (all training + eval code)
    - `README.md` (approach description)
    - `eval/eval.sh` (evaluation script)
    - Log files or placeholders for 3-seed runs

18. **eval.sh correctness:** The eval script should invoke `torchrun --standalone --nproc_per_node=8 train_gpt.py` with appropriate flags. It should accept `SEED` as an environment variable.

### E. RUNTIME FEASIBILITY

19. **Training time:** With `MAX_WALLCLOCK_SECONDS=600` and `ITERATIONS=20000` (capped by time), verify that the training loop has a wallclock check and exits cleanly when time is up. It should save the best model, not crash.

20. **Eval time:** The author's benchmark extrapolates to ~503s on 8xH100 with stride=128, without TTT. Verify there are no hidden O(n²) loops or per-token Python overhead that would scale poorly to 62M tokens.

21. **Memory:** 29.8M params at bf16 = ~60MB. On H100 (80GB), this is fine. But check that eval doesn't accumulate tensors without freeing them (e.g., storing all logits in a list).

### F. CODE QUALITY AND RISKS

22. **Consistency between train_gpt_phase3.py and the submission's train_gpt.py:** These should be functionally identical. Flag any differences.

23. **Hardcoded paths:** The code must NOT have hardcoded absolute paths (e.g., `C:/Users/...`). All paths should be relative to the script location or configurable via env vars.

24. **Seed handling:** `SEED` env var should control all sources of randomness (torch, numpy, python random). Different seeds should produce different but valid results.

25. **Error handling:** What happens if training is interrupted at step 50? Does the code save a partial checkpoint? What if eval runs out of time — does it report partial BPB or crash?

---

## Expected output format

Organize your review as:

```
## CRITICAL ISSUES (must fix before cloud)
- [Issue description, file, line number if possible, why it matters]

## WARNINGS (should fix, risk of problems)
- [Issue description, risk level, suggested fix]

## VERIFIED OK
- [Each checklist item that passes, with brief evidence]

## RECOMMENDATIONS
- [Optional improvements, not blocking]
```

Focus especially on **Section A (Legality)** — a single violation there means disqualification regardless of BPB score. For each legality check, quote the specific code that proves compliance or cite the specific code that violates it.
