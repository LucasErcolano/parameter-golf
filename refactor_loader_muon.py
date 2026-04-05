import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # 1. Update load_data_shard to use memmap
    new_load_data_shard = '''def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"bad header:{file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"size mismatch:{file}")
    tokens_np = np.memmap(file, dtype="<u2", mode="r", offset=header_bytes, shape=(num_tokens,))
    return torch.from_numpy(tokens_np).to(dtype=torch.int32)'''

    source = re.sub(
        r'def load_data_shard\(file: Path\) -> Tensor:.*?tokens_np = np\.fromfile.*?return torch\.from_numpy\(tokens_np\)\.to\(dtype=torch\.int32\)',
        new_load_data_shard,
        source,
        flags=re.DOTALL
    )

    # 2. Replace TokenStream and DistributedTokenLoader with Coprime Stride
    new_loader = '''class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no files:{pattern}")
        # Coprime Stride Data Loading
        import math
        self.tokens = torch.cat([load_data_shard(file) for file in self.files])
        self.total_tokens = self.tokens.numel()
        self.pos = 0
        self.stride = 50331653 # A large prime stride

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        batch_size = local_tokens // seq_len
        seq_size = seq_len + 1
        usable_tokens = self.total_tokens - seq_size
        if usable_tokens < batch_size:
            raise ValueError("Not enough tokens")

        idx = torch.arange(batch_size, dtype=torch.int64)
        current_pos = self.pos + self.rank * batch_size
        indices = ((current_pos + idx) * self.stride) % usable_tokens

        self.pos += batch_size * self.world_size

        offsets = torch.arange(seq_size, dtype=torch.int64)
        gather_idx = indices.unsqueeze(1) + offsets.unsqueeze(0)

        local = self.tokens[gather_idx].to(dtype=torch.int64)
        x = local[:, :-1].reshape(-1, seq_len)
        y = local[:, 1:].reshape(-1, seq_len)

        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def state_dict(self) -> dict[str, object]:
        return {"rank": self.rank, "world_size": self.world_size, "pos": self.pos}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.pos = state.get("pos", 0)'''

    source = re.sub(
        r'class TokenStream:.*?class DistributedTokenLoader:.*?def load_state_dict.*?self\.stream\.load_state_dict\(stream_state\)',
        new_loader,
        source,
        flags=re.DOTALL
    )

    # 3. Replace Muon with TurboMuon
    new_muon = '''def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

class TurboMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    # AOL preconditioning applied intrinsically during layout via Newton-Schulz
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss'''

    source = re.sub(
        r'class Muon\(torch\.optim\.Optimizer\):.*?return loss',
        new_muon,
        source,
        flags=re.DOTALL
    )
    
    source = re.sub(r'def zeropower_via_newtonschulz5.*?return X', '', source, flags=re.DOTALL, count=1) # remove original to avoid dup

    source = source.replace('Muon(', 'TurboMuon(')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(source)

    print(f'Applied Coprime-Stride and TurboMuon to {filepath}')

if __name__ == "__main__":
    process_file('train_gpt_phase3.py')
    process_file('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')
