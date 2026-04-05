from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def url(self) -> str:
        return f"{GRAPHQL_URL}?api_key={self.api_key}"

    def rest_request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{REST_URL}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RunPodError(exc.read().decode("utf-8", errors="replace")) from exc
        return payload

    def request(self, query: str, variables: dict | None = None) -> dict:
        body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RunPodError(exc.read().decode("utf-8", errors="replace")) from exc
        if payload.get("errors"):
            raise RunPodError(json.dumps(payload["errors"], indent=2))
        return payload["data"]

    def get_pod(self, pod_id: str) -> dict:
        return self.rest_request("GET", f"/pods/{pod_id}")

    def resume_spot(self, pod_id: str, gpu_count: int, bid_per_gpu: float) -> dict:
        # REST start/resume does not expose bid tuning here; for current Spot workflow,
        # pods are created with Spot pricing and resumed as-is.
        return self.rest_request("POST", f"/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> dict:
        return self.rest_request("POST", f"/pods/{pod_id}/stop")


def run_subprocess(
    args: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        timeout=timeout,
        text=True,
        capture_output=capture,
    )


def ssh_args(key_path: Path, port: int, host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        str(port),
        f"root@{host}",
    ]


def scp_args(key_path: Path, port: int) -> list[str]:
    return [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-P",
        str(port),
    ]


def ssh(
    key_path: Path,
    host: str,
    port: int,
    command: str,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_subprocess(
        [*ssh_args(key_path, port, host), command],
        check=check,
        timeout=timeout,
    )


def scp_from(
    key_path: Path,
    host: str,
    port: int,
    remote_path: str,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        [*scp_args(key_path, port), f"root@{host}:{remote_path}", str(local_path)],
        check=True,
        timeout=300,
    )


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def extract_public_ssh(runtime: dict | None) -> tuple[str, int] | None:
    if not runtime:
        return None
    if runtime.get("publicIp") and runtime.get("portMappings"):
        port = runtime["portMappings"].get("22")
        if port:
            return runtime["publicIp"], int(port)
    for port_info in runtime.get("ports") or []:
        if not isinstance(port_info, dict):
            continue
        if port_info.get("type") == "tcp" and int(port_info.get("privatePort", 0)) == 22 and port_info.get("isIpPublic"):
            return port_info["ip"], int(port_info["publicPort"])
    return None


def wait_for_ssh(
    client: RunPodClient,
    pod_id: str,
    timeout_seconds: float,
) -> tuple[str, int]:
    start = time.perf_counter()
    while True:
        pod = client.get_pod(pod_id)
        ssh_target = extract_public_ssh(pod)
        if ssh_target is not None:
            return ssh_target
        if time.perf_counter() - start >= timeout_seconds:
            raise RunPodError(f"Timed out waiting for SSH on pod {pod_id}")
        time.sleep(10)


def parse_step_metrics(log_text: str) -> tuple[int, float] | None:
    step_matches = [int(m) for m in re.findall(r"s:(\d+)(?:/\d+)?", log_text)]
    sa_matches = [float(m) for m in re.findall(r"sa:([0-9.]+)ms", log_text)]
    if not step_matches or not sa_matches:
        return None
    # Use the maximum observed step so checkpoint lines like
    # "[spot] ckpt:... s:755" count as progress even if the latest
    # structured training log has not flushed yet.
    return max(step_matches), sa_matches[-1]


def parse_allreduce_ms(text: str) -> float | None:
    m = re.search(r"ALLREDUCE_MS_AVG=([0-9.]+)", text)
    if not m:
        return None
    return float(m.group(1))


def ensure_remote_dir(key_path: Path, host: str, port: int, path: str) -> None:
    ssh(key_path, host, port, f"mkdir -p {shlex.quote(path)}", timeout=60)


def verify_remote_volume_ready(
    key_path: Path,
    host: str,
    port: int,
    *,
    remote_workdir: str,
    dataset_dir: str,
) -> None:
    cmd = textwrap.dedent(
        f"""
        set -euo pipefail
        test -f {shlex.quote(remote_workdir + "/train_gpt.py")}
        test -d {shlex.quote(dataset_dir)}
        test "$(find {shlex.quote(dataset_dir)} -mindepth 1 -maxdepth 1 | wc -l)" -gt 0
        """
    ).strip()
    try:
        ssh(key_path, host, port, cmd, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise RunPodError(
            "Remote volume is not ready: expected record workdir and pre-populated dataset to exist. "
            "Seed the volume on a cheap/transfer pod before using H100 Spot."
        ) from exc


def run_remote_preflight(
    key_path: Path,
    host: str,
    port: int,
    *,
    gpu_count: int,
    max_allreduce_ms: float,
    allow_sys_topology: bool,
) -> None:
    log("Running CUDA sanity check")
    sanity_cmd = textwrap.dedent(
        f"""
        python3 - <<'PY'
        import torch
        assert torch.cuda.is_available(), "CUDA unavailable"
        n = torch.cuda.device_count()
        print(f"CUDA_DEVICE_COUNT={{n}}")
        assert n >= {gpu_count}, f"need at least {gpu_count} GPUs, got {{n}}"
        names = [torch.cuda.get_device_name(i) for i in range({gpu_count})]
        print("CUDA_DEVICES=" + ",".join(names))
        # Fail fast on flaky GPUs before paying the distributed startup cost.
        checksums = []
        for i in range({gpu_count}):
            torch.cuda.set_device(i)
            a = torch.randn((2048, 2048), device=f"cuda:{{i}}", dtype=torch.float16)
            b = torch.randn((2048, 2048), device=f"cuda:{{i}}", dtype=torch.float16)
            c = None
            for _ in range(6):
                c = a @ b
            torch.cuda.synchronize(i)
            checksums.append(float(c[0, 0].item()))
        print("CUDA_BURNIN_OK=" + ",".join(f"{{x:.4f}}" for x in checksums))
        PY
        """
    ).strip()
    sanity = ssh(key_path, host, port, sanity_cmd, timeout=120)
    log(sanity.stdout.strip())

    log("Collecting topology")
    topo = ssh(key_path, host, port, "nvidia-smi topo -m", timeout=120)
    topo_text = topo.stdout
    if not allow_sys_topology:
        gpu_rows = [line for line in topo_text.splitlines() if line.startswith("GPU")]
        bad_links = [line for line in gpu_rows if " SYS " in f" {line} " or " NODE " in f" {line} "]
        if bad_links:
            raise RunPodError("GPU topology shows SYS/NODE links; treating pod as fractured/lemon")

    log("Collecting NUMA layout")
    try:
        numa = ssh(key_path, host, port, "numactl -H", timeout=120)
        log(numa.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise RunPodError("numactl unavailable; treating pod as invalid for fail-fast preflight") from exc

    log("Running NCCL all-reduce probe")
    nccl_cmd = textwrap.dedent(
        f"""
        cat > /tmp/runpod_nccl_probe.py <<'PY'
        import os
        import time
        import torch
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        n = (8 * 1024 * 1024) // 2
        x = torch.ones(n, device="cuda", dtype=torch.bfloat16)
        for _ in range(10):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        if rank == 0:
            times_sorted = sorted(times)
            p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
            print(f"ALLREDUCE_MS_AVG={{sum(times)/len(times):.3f}}")
            print(f"ALLREDUCE_MS_P95={{p95:.3f}}")
        dist.destroy_process_group()
        PY
        cd /tmp && NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node={gpu_count} /tmp/runpod_nccl_probe.py
        """
    ).strip()
    nccl = ssh(key_path, host, port, nccl_cmd, timeout=300)
    allreduce_ms = parse_allreduce_ms(nccl.stdout)
    log(nccl.stdout.strip())
    if allreduce_ms is None:
        raise RunPodError("NCCL probe did not emit ALLREDUCE_MS_AVG")
    if allreduce_ms > max_allreduce_ms:
        raise RunPodError(
            f"NCCL all-reduce latency too high: {allreduce_ms:.3f} ms > {max_allreduce_ms:.3f} ms"
        )


def launch_remote_seed(
    key_path: Path,
    host: str,
    port: int,
    *,
    remote_workdir: str,
    remote_log_dir: str,
    remote_ckpt_dir: str,
    seed: int,
    nproc_per_node: int,
    eval_stride: int,
    eval_batch_seqs: int,
    ttt_enabled: int,
    eval_timeout_seconds: float,
    max_wallclock_seconds: float,
    ckpt_every_secs: float,
    ckpt_every_steps: int,
) -> int:
    ensure_remote_dir(key_path, host, port, remote_log_dir)
    ensure_remote_dir(key_path, host, port, remote_ckpt_dir)
    train_log = f"{remote_log_dir}/seed_{seed}_train_stdout.log"
    eval_log = f"{remote_log_dir}/seed_{seed}_eval_stdout.log"
    batch_log = f"{remote_log_dir}/batch.log"
    remote_script = textwrap.dedent(
        f"""
        cd {shlex.quote(remote_workdir)}
        mkdir -p {shlex.quote(remote_log_dir)} {shlex.quote(remote_ckpt_dir)}
        echo "=== SEED {seed} TRAIN START $(date -Is) ===" >> {shlex.quote(batch_log)}
        export SEED={seed}
        export RUN_ID=cloud_seed{seed}_planner
        export USE_LIBUV=0
        export CKPT_DIR={shlex.quote(remote_ckpt_dir)}
        export CKPT_EVERY_SECS={ckpt_every_secs}
        export CKPT_EVERY_STEPS={ckpt_every_steps}
        export RESUME_CKPT=1
        export TTT_ENABLED={ttt_enabled}
        export EVAL_STRIDE={eval_stride}
        export EVAL_BATCH_SEQS={eval_batch_seqs}
        export TTT_BATCH_SEQS={eval_batch_seqs}
        export TTT_TRAIN_BATCH_SEQS=8
        export LEMON_STEP=0
        export EVAL_TIMEOUT_SECONDS={eval_timeout_seconds}
        export MAX_WALLCLOCK_SECONDS={max_wallclock_seconds}
        torchrun --standalone --nproc_per_node={nproc_per_node} train_gpt.py 2>&1 | tee {shlex.quote(train_log)}
        echo "=== SEED {seed} EVAL START $(date -Is) ===" >> {shlex.quote(batch_log)}
        EVAL_ONLY=1 CHECKPOINT_PATH=final_model.pt bash eval/eval.sh 2>&1 | tee {shlex.quote(eval_log)}
        echo "=== SEED {seed} DONE $(date -Is) ===" >> {shlex.quote(batch_log)}
        """
    ).strip()
    wrapped = f"nohup bash -lc {shlex.quote(remote_script)} > {shlex.quote(remote_log_dir + '/nohup.out')} 2>&1 & echo $!"
    resp = ssh(key_path, host, port, wrapped, timeout=120)
    pid_text = resp.stdout.strip().splitlines()[-1]
    return int(pid_text)


def remote_pid_alive(key_path: Path, host: str, port: int, pid: int) -> bool:
    resp = ssh(
        key_path,
        host,
        port,
        f"ps -p {pid} -o pid= || true",
        timeout=60,
    )
    return bool(resp.stdout.strip())


def download_logs(
    key_path: Path,
    host: str,
    port: int,
    *,
    remote_workdir: str,
    remote_log_dir: str,
    local_dir: Path,
    seed: int,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        f"{remote_workdir}/logs/cloud_seed{seed}_planner.txt": local_dir / f"cloud_seed{seed}_planner.txt",
        f"{remote_log_dir}/seed_{seed}_train_stdout.log": local_dir / f"seed_{seed}_train_stdout.log",
        f"{remote_log_dir}/seed_{seed}_eval_stdout.log": local_dir / f"seed_{seed}_eval_stdout.log",
        f"{remote_log_dir}/batch.log": local_dir / "batch.log",
        f"{remote_log_dir}/nohup.out": local_dir / "nohup.out",
    }
    for remote_path, local_path in candidates.items():
        try:
            scp_from(key_path, host, port, remote_path, local_path)
        except subprocess.CalledProcessError:
            pass


def stop_pod_or_raise(client: RunPodClient, pod_id: str, reason: str) -> None:
    log(f"Stopping pod {pod_id}: {reason}")
    client.stop_pod(pod_id)
    raise RunPodError(reason)


def monitor_run(
    client: RunPodClient,
    key_path: Path,
    host: str,
    port: int,
    *,
    pod_id: str,
    pid: int,
    remote_log_dir: str,
    remote_workdir: str,
    local_download_dir: Path,
    seed: int,
    failfast_grace_seconds: float,
    failfast_min_step: int,
    failfast_max_sa_ms: float,
    stop_on_fail: bool,
) -> None:
    start = time.perf_counter()
    train_log = f"{remote_log_dir}/seed_{seed}_train_stdout.log"
    last_metrics: tuple[int, float] | None = None
    while True:
        alive = remote_pid_alive(key_path, host, port, pid)
        tail = ssh(
            key_path,
            host,
            port,
            f"tail -n 200 {shlex.quote(train_log)} 2>/dev/null || true",
            timeout=120,
        ).stdout
        metrics = parse_step_metrics(tail)
        if metrics is not None:
            last_metrics = metrics
        elapsed = time.perf_counter() - start
        if elapsed >= failfast_grace_seconds and last_metrics is not None:
            step, sa_ms = last_metrics
            slow_step = step < failfast_min_step
            slow_sa = sa_ms > failfast_max_sa_ms
            # Treat bad step count as advisory until a second grace window elapses,
            # because training logs flush sparsely and checkpoints may be the most
            # recent step telemetry available.
            if slow_sa or (slow_step and elapsed >= failfast_grace_seconds * 2):
                download_logs(
                    key_path,
                    host,
                    port,
                    remote_workdir=remote_workdir,
                    remote_log_dir=remote_log_dir,
                    local_dir=local_download_dir,
                    seed=seed,
                )
                msg = (
                    f"Fail-fast triggered for seed {seed}: step={step}, sa_ms={sa_ms:.2f}, "
                    f"grace={failfast_grace_seconds}s"
                )
                if stop_on_fail:
                    stop_pod_or_raise(client, pod_id, msg)
                raise RunPodError(msg)
            if slow_step and not slow_sa:
                log(
                    f"Fail-fast advisory for seed {seed}: step={step} below target "
                    f"{failfast_min_step} at {elapsed:.0f}s, but sa_ms={sa_ms:.2f} looks healthy; "
                    "continuing until next grace window"
                )
        if elapsed >= failfast_grace_seconds and last_metrics is None:
            download_logs(
                key_path,
                host,
                port,
                remote_workdir=remote_workdir,
                remote_log_dir=remote_log_dir,
                local_dir=local_download_dir,
                seed=seed,
            )
            msg = f"Fail-fast triggered for seed {seed}: no parsed step telemetry after {failfast_grace_seconds:.0f}s"
            if stop_on_fail:
                stop_pod_or_raise(client, pod_id, msg)
            raise RunPodError(msg)
        if not alive:
            break
        time.sleep(20)

    download_logs(
        key_path,
        host,
        port,
        remote_workdir=remote_workdir,
        remote_log_dir=remote_log_dir,
        local_dir=local_download_dir,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunPod Spot planner with fail-fast preflight and telemetry guards")
    p.add_argument("--pod-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    p.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    p.add_argument("--resume-if-needed", action="store_true")
    p.add_argument("--gpu-count", type=int, default=8)
    p.add_argument("--bid-per-gpu", type=float, default=2.0)
    p.add_argument(
        "--remote-workdir",
        default="/workspace/parameter-golf/records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram",
    )
    p.add_argument("--remote-log-dir", default="/workspace/run_logs/spot_planner")
    p.add_argument("--remote-ckpt-dir", default="/workspace/checkpoints/spot_planner")
    p.add_argument(
        "--local-download-dir",
        default=str(Path.cwd() / "logs" / "runpod_spot_planner"),
    )
    p.add_argument(
        "--dataset-dir",
        default="/workspace/parameter-golf/data/datasets/fineweb10B_sp1024",
    )
    p.add_argument("--skip-volume-ready-check", action="store_true")
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--eval-stride", type=int, default=256)
    p.add_argument("--eval-batch-seqs", type=int, default=32)
    p.add_argument("--ttt-enabled", type=int, default=0)
    p.add_argument("--eval-timeout-seconds", type=float, default=590.0)
    p.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    p.add_argument("--ckpt-every-secs", type=float, default=60.0)
    p.add_argument("--ckpt-every-steps", type=int, default=500)
    p.add_argument("--preflight-timeout-seconds", type=float, default=600.0)
    p.add_argument("--preflight-max-allreduce-ms", type=float, default=12.0)
    p.add_argument("--allow-sys-topology", action="store_true")
    p.add_argument("--failfast-grace-seconds", type=float, default=180.0)
    p.add_argument("--failfast-min-step", type=int, default=1200)
    p.add_argument("--failfast-max-sa-ms", type=float, default=180.0)
    p.add_argument("--no-stop-on-fail", action="store_true")
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
        log(f"Resuming spot pod {args.pod_id}")
        client.resume_spot(args.pod_id, args.gpu_count, args.bid_per_gpu)

    host, port = wait_for_ssh(client, args.pod_id, args.preflight_timeout_seconds)
    key_path = Path(args.ssh_key).expanduser().resolve()
    local_download_dir = Path(args.local_download_dir).expanduser().resolve()

    log(f"SSH ready at {host}:{port}")
    try:
        if not args.skip_volume_ready_check:
            log("Checking pre-populated volume readiness")
            verify_remote_volume_ready(
                key_path,
                host,
                port,
                remote_workdir=args.remote_workdir,
                dataset_dir=args.dataset_dir,
            )
        run_remote_preflight(
            key_path,
            host,
            port,
            gpu_count=args.gpu_count,
            max_allreduce_ms=args.preflight_max_allreduce_ms,
            allow_sys_topology=args.allow_sys_topology,
        )
        pid = launch_remote_seed(
            key_path,
            host,
            port,
            remote_workdir=args.remote_workdir,
            remote_log_dir=args.remote_log_dir,
            remote_ckpt_dir=args.remote_ckpt_dir,
            seed=args.seed,
            nproc_per_node=args.nproc_per_node,
            eval_stride=args.eval_stride,
            eval_batch_seqs=args.eval_batch_seqs,
            ttt_enabled=args.ttt_enabled,
            eval_timeout_seconds=args.eval_timeout_seconds,
            max_wallclock_seconds=args.max_wallclock_seconds,
            ckpt_every_secs=args.ckpt_every_secs,
            ckpt_every_steps=args.ckpt_every_steps,
        )
        log(f"Launched remote seed {args.seed} with PID {pid}")
        monitor_run(
            client,
            key_path,
            host,
            port,
            pod_id=args.pod_id,
            pid=pid,
            remote_log_dir=args.remote_log_dir,
            remote_workdir=args.remote_workdir,
            local_download_dir=local_download_dir,
            seed=args.seed,
            failfast_grace_seconds=args.failfast_grace_seconds,
            failfast_min_step=args.failfast_min_step,
            failfast_max_sa_ms=args.failfast_max_sa_ms,
            stop_on_fail=not args.no_stop_on_fail,
        )
        log(f"Seed {args.seed} finished; logs downloaded to {local_download_dir}")
    except Exception:
        if not args.no_stop_on_fail:
            try:
                client.stop_pod(args.pod_id)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
