from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import textwrap
from pathlib import Path

from runpod_spot_planner import RunPodClient, RunPodError, log, scp_args, ssh, wait_for_ssh


def load_hf_token(env_name: str, env_file: Path) -> str:
    token = os.environ.get(env_name)
    if token:
        return token
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(f"{env_name}="):
                return line.split("=", 1)[1]
    raise RunPodError(f"Missing {env_name} in environment or {env_file}")


def scp_to_dir(key_path: Path, host: str, port: int, local_path: Path, remote_path: str) -> None:
    subprocess.run(
        [*scp_args(key_path, port), "-r", str(local_path), f"root@{host}:{remote_path}"],
        check=True,
        timeout=3600,
        text=True,
        capture_output=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-populate a RunPod volume on a cheap/transfer pod")
    p.add_argument("--pod-id", required=True)
    p.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    p.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    p.add_argument("--resume-if-needed", action="store_true")
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--bid-per-gpu", type=float, default=2.0)
    p.add_argument("--remote-repo-dir", default="/workspace/parameter-golf")
    p.add_argument(
        "--record-src",
        default=str(
            Path.cwd()
            / "records"
            / "track_10min_16mb"
            / "2026-04-04_LucasErcolano_MixedQuantNgram"
        ),
    )
    p.add_argument(
        "--record-name",
        default="2026-04-04_LucasErcolano_MixedQuantNgram",
    )
    p.add_argument("--train-shards", type=int, default=80)
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--env-file", default=str(Path.cwd() / ".env"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in env var {args.api_key_env}")
    client = RunPodClient(api_key)
    pod = client.get_pod(args.pod_id)
    if pod["desiredStatus"] != "RUNNING":
        if not args.resume_if_needed:
            raise SystemExit(f"Pod {args.pod_id} is {pod['desiredStatus']}; pass --resume-if-needed or start it manually")
        log(f"Resuming pod {args.pod_id}")
        client.resume_spot(args.pod_id, args.gpu_count, args.bid_per_gpu)
    host, port = wait_for_ssh(client, args.pod_id, 900)
    key_path = Path(args.ssh_key).expanduser().resolve()
    record_src = Path(args.record_src).expanduser().resolve()
    hf_token = load_hf_token(args.hf_token_env, Path(args.env_file).expanduser().resolve())

    log(f"Seeding volume on {host}:{port}")
    ssh(key_path, host, port, "mkdir -p /workspace/record_sync_tmp", timeout=60)
    scp_to_dir(key_path, host, port, record_src, "/workspace/record_sync_tmp")

    remote_record_tmp = f"/workspace/record_sync_tmp/{record_src.name}"
    remote_record_final = f"{args.remote_repo_dir}/records/track_10min_16mb/{args.record_name}"
    cmd = textwrap.dedent(
        f"""
        set -euo pipefail
        export HF_TOKEN={shlex.quote(hf_token)}
        export HF_HOME=/workspace/hf_home
        cd /workspace
        if [ ! -d {shlex.quote(args.remote_repo_dir + '/.git')} ]; then
          git clone https://github.com/openai/parameter-golf.git {shlex.quote(args.remote_repo_dir)}
        fi
        cd {shlex.quote(args.remote_repo_dir)}
        python3 -m pip install -q --break-system-packages sentencepiece zstandard huggingface_hub tqdm
        mkdir -p records/track_10min_16mb
        rm -rf {shlex.quote(remote_record_final)}
        cp -r {shlex.quote(remote_record_tmp)} {shlex.quote(remote_record_final)}
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards {args.train_shards}
        python3 -m py_compile {shlex.quote(remote_record_final + '/train_gpt.py')}
        du -sh data/datasets/fineweb10B_sp1024
        """
    ).strip()
    out = ssh(key_path, host, port, cmd, timeout=7200)
    if out.stdout.strip():
        print(out.stdout)
    if out.stderr.strip():
        print(out.stderr)
    log("Volume seed complete")


if __name__ == "__main__":
    main()
