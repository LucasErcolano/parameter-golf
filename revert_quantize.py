import re

def process(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_func = '''def quantize_signed_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    clip_range = (1 << (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        best_q = torch.zeros_like(t32, dtype=torch.int8)
        best_s = torch.zeros(t32.shape[0], dtype=torch.float16, device=t.device)
        best_err = torch.full((t32.shape[0],), float('inf'), device=t.device)
        
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean(dim=1)
            
            mask = err < best_err
            best_err[mask] = err[mask]
            best_s[mask] = s[mask]
            best_q[mask] = q[mask]
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale'''

    old_func = '''def quantize_signed_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    clip_range = (1 << (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale'''

    content = content.replace(new_func, old_func)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done revert", filepath)

process('train_gpt_phase3.py')
process('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')