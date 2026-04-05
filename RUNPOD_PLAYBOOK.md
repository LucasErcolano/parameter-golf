# RunPod Playbook

## Package

- Working directory:
  - `records/track_10min_16mb/2026-04-03_PRAligned_ComplementaryBackoff_LoRATTT_3090`
- First cloud run target:
  - `EVAL_STRIDE=256`
  - `EVAL_BATCH_SEQS=32`
  - `TTT_ENABLED=0`

## One-Time Setup

```bash
cd /workspace
git clone <YOUR_FORK_OR_REPO_URL> parameter-golf
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
mkdir -p /workspace/checkpoints
cd records/track_10min_16mb/2026-04-03_PRAligned_ComplementaryBackoff_LoRATTT_3090
```

## Volume-First Workflow

- Do **not** spend H100 time downloading/tokenizing data.
- Seed `/workspace` on a cheap CPU / transfer pod first, then attach that pre-populated volume to the H100 Spot pod.
- Use [runpod_volume_seed.py](C:/Users/joaco/Documents/IA/Year-4/Semestre-1/NLP/parameter-golf-main/runpod_volume_seed.py) to prepare:
  - the repo clone
  - the record folder
  - `fineweb10B_sp1024`

Example:

```bash
export RUNPOD_API_KEY=...
python3 runpod_volume_seed.py \
  --pod-id <TRANSFER_OR_CHEAP_POD_ID> \
  --resume-if-needed \
  --record-name 2026-04-04_LucasErcolano_MixedQuantNgram
```

- The H100 planner now expects the volume to already be ready.
- If the dataset or record folder is missing, the planner fails immediately instead of doing I/O on the GPU box.

## Spot Resume

- Training now checkpoints to `CKPT_DIR/train_ckpt_seed<SEED>.pt`.
- Defaults:
  - `CKPT_DIR=/workspace/checkpoints` when `/workspace` exists
  - `CKPT_EVERY_SECS=60`
  - `CKPT_EVERY_STEPS=500` when driven by the central planner
  - `RESUME_CKPT=1`
- If a Spot interruption happens, relaunch the pod and rerun the exact same training command.
- Final artifacts are copied to `CKPT_DIR/artifact_seed<SEED>/`.

## Central Planner

- Use [runpod_spot_planner.py](C:/Users/joaco/Documents/IA/Year-4/Semestre-1/NLP/parameter-golf-main/runpod_spot_planner.py) for Spot runs that should fail fast on bad hosts.
- The planner does:
  - volume readiness check
  - CUDA sanity
  - `nvidia-smi topo -m` capture
  - NUMA capture via `numactl -H`
  - NCCL all-reduce latency probe
  - early telemetry kill if training stays too slow after the grace window
  - log download on finish or fail
  - pod stop on lemon detection

Example:

```bash
export RUNPOD_API_KEY=...
python3 runpod_spot_planner.py \
  --pod-id <RUNPOD_POD_ID> \
  --resume-if-needed \
  --seed 42 \
  --remote-workdir /workspace/parameter-golf/records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram \
  --dataset-dir /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  --remote-log-dir /workspace/run_logs/spot_planner \
  --remote-ckpt-dir /workspace/checkpoints/spot_planner
```

Recommended fail-fast defaults:

- `--preflight-max-allreduce-ms 12`
- `--failfast-grace-seconds 180`
- `--failfast-min-step 1200`
- `--failfast-max-sa-ms 180`

## Run 1

```bash
SEED=42 \
USE_LIBUV=0 \
CKPT_DIR=/workspace/checkpoints \
torchrun --standalone --nproc_per_node=8 train_gpt.py

SEED=42 \
USE_LIBUV=0 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=32 \
TTT_ENABLED=0 \
bash eval/eval.sh
```

Record:
- Train wallclock:
- Eval wallclock:
- Final BPB:

## Run 2

```bash
SEED=1337 \
USE_LIBUV=0 \
CKPT_DIR=/workspace/checkpoints \
torchrun --standalone --nproc_per_node=8 train_gpt.py

SEED=1337 \
USE_LIBUV=0 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=32 \
TTT_ENABLED=0 \
bash eval/eval.sh
```

Record:
- Train wallclock:
- Eval wallclock:
- Final BPB:

## Run 3

```bash
SEED=7 \
USE_LIBUV=0 \
CKPT_DIR=/workspace/checkpoints \
torchrun --standalone --nproc_per_node=8 train_gpt.py

SEED=7 \
USE_LIBUV=0 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=32 \
TTT_ENABLED=0 \
bash eval/eval.sh
```

Record:
- Train wallclock:
- Eval wallclock:
- Final BPB:

## Decision Tree

- If `BPB < 0.50`: keep the package and submit.
- If `0.50 <= BPB < 0.80`: keep the training config and test `EVAL_STRIDE=512` or a TTT eval pass.
- If `BPB >= 0.80`: inspect logs first; assume something is broken before tuning.
- If training times out: lower `ITERATIONS` or `TRAIN_BATCH_TOKENS`.
- If eval times out: raise `EVAL_STRIDE` to `512`.

## Notes

- The current defaults already target:
  - `11L / 512d / MLP 3.5`
  - `XSA-4`
  - `VE128`
  - `VRL`
  - `LeakyReLU^2`
  - `EMA + SWA`
  - `mixed int6 + lzma`
  - causal `BackoffNgramMixer`
- First cloud pass should stay `ngram-only`.
- Re-test `TTT_ENABLED=1` only after a strong full training run shows a measurable gain.
- If you expose a web UI or live dashboard and proxy latency is severe, prefer exposing a raw `tcp` port instead of `http` so you bypass the platform reverse proxy.
