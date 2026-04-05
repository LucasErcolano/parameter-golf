import subprocess
import sys
import os
import time
import re

def stream_command(cmd, env):
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    process.wait()
    return "".join(lines)

env = os.environ.copy()
env["TORCH_COMPILE"] = "0"
env["PYTHONUNBUFFERED"] = "1"

print("--- Check 3: Determinism ---")
env_det = env.copy()
env_det["SEED"] = "42"
env_det["ITERATIONS"] = "15"
out1 = stream_command([sys.executable, "train_gpt.py"], env_det)
out2 = stream_command([sys.executable, "train_gpt.py"], env_det)

def extract_losses(out):
    losses = []
    for line in out.splitlines():
        if "train_loss:" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("train_loss:"):
                    losses.append(p.split(":")[1])
    return losses

losses1 = extract_losses(out1)
losses2 = extract_losses(out2)
if losses1 and losses1 == losses2:
    print(f"CHECK 3 PASSED: Determinism OK. Losses: {losses1}")
else:
    print(f"CHECK 3 FAILED: {losses1} != {losses2}")

print("--- Check 1, 2, 4, 7, 8: Full 200 Steps Run ---")
env_main = env.copy()
env_main["ITERATIONS"] = "200"
env_main["VAL_LOSS_EVERY"] = "200"
env_main["GRAD_CLIP_NORM"] = "1.0"
out_main = stream_command([sys.executable, "train_gpt.py"], env_main)

print('--- Parsing Results ---')
val_bpb = None
final_q_bpb = None
final_q_bpb_exact = None
artifact_size = 0
for line in out_main.splitlines():
    if 'val_loss:' in line and 'step:200/200' in line:
        m = re.search(r'val_bpb:([\d\.]+)', line)
        if m: val_bpb = float(m.group(1))
    if 'final_int8_zlib_roundtrip val_loss:' in line:
        m = re.search(r'val_bpb:([\d\.]+)', line)
        if m: final_q_bpb = float(m.group(1))
    if 'final_int8_zlib_roundtrip_exact val_loss:' in line:
        m = re.search(r'val_bpb:([\d\.]+)', line)
        if m: final_q_bpb_exact = float(m.group(1))

print(f'Train val_bpb: {val_bpb}')
print(f'Quant val_bpb: {final_q_bpb}')
if val_bpb is not None and final_q_bpb is not None:
    gap = final_q_bpb - val_bpb
    print(f'CHECK 1: Quant gap = {gap:.5f} (Target < 0.003: {gap < 0.003})')
    print(f'CHECK 4: Eval reproduce train: Match? {abs(val_bpb - final_q_bpb) < 0.0001}')

if os.path.exists('final_model.int8.ptz'):
    size_mb = os.path.getsize('final_model.int8.ptz') / (1024*1024)
    print(f'CHECK 2: Artifact size = {size_mb:.2f} MB (Target < 15.5: {size_mb < 15.5})')
