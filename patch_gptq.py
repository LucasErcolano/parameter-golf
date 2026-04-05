import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    gptq_function = '''
def gptq_quantize_tensor(W_tensor, H_tensor, bits):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = W_tensor.clone().float().to(device)
    H = H_tensor.float().to(device)
    out_features, in_features = W.shape
    clip_range = (1 << (bits - 1)) - 1
    
    # Compute scales per row based on original W
    row_clip = W.abs().amax(dim=1)
    scale = (row_clip / clip_range).clamp_min(1e-8)
    
    damp = 0.01
    diag = torch.diag(H)
    damp_val = damp * torch.mean(diag)
    H_inv = torch.linalg.inv(H + (damp_val + 1e-4) * torch.eye(in_features, device=device))
    
    block_size = 128
    
    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        count = i2 - i1
        
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        H_inv1 = H_inv[i1:i2, i1:i2]
        
        for i in range(count):
            w = W1[:, i]
            d = H_inv1[i, i]
            
            q = torch.clamp(torch.round(w / scale), -clip_range, clip_range) * scale
            Q1[:, i] = q
            err = (w - q) / d
            Err1[:, i] = err
            W1[:, i:] -= err.unsqueeze(1) * H_inv1[i, i:].unsqueeze(0)
            
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ H_inv[i1:i2, i2:]
        
    Q_final = torch.clamp(torch.round(W / scale[:, None]), -clip_range, clip_range).to(torch.int8)
    return Q_final.cpu(), scale.to(torch.float16).cpu()

'''

    if 'def gptq_quantize_tensor' not in source:
        source = source.replace('def mixed_quantize_int6', gptq_function + 'def mixed_quantize_int6')

    quant_block = '''        bits = int(quant_bits_by_cat.get(cat, 0))
        if bits in (5, 6) and t.ndim >= 1:
            q, s = quantize_signed_per_row(t, bits)
            q_packed, q_numel = pack_signed_tensor(q, bits)'''
            
    quant_block_new = '''        bits = int(quant_bits_by_cat.get(cat, 0))
        if bits in (5, 6) and t.ndim >= 1:
            H_dict = state_dict.get("_h_dict", {})
            if name in H_dict:
                q, s = gptq_quantize_tensor(t, H_dict[name], bits)
            else:
                q, s = quantize_signed_per_row(t, bits)
            q_packed, q_numel = pack_signed_tensor(q, bits)'''

    source = source.replace(quant_block, quant_block_new)

    # In training loop:
    old_h_code = '''        for name, H in H_dict.items():
            diag = torch.diag(H)
            damp_val = damp * torch.mean(diag)
            H_inv = torch.linalg.inv(H + (damp_val + 1e-4) * torch.eye(H.size(0), device=H.device))
            # Simulated GPTQ weight update (in practice, this updates m.weight using H_inv)
            # We keep the model weights updated for artifact packaging.
            pass
        log0("Full Hessian GPTQ Calibration complete.")'''

    new_h_code = '''        log0("Saving H_dict for GPTQ in eval...")
        import os
        torch.save({k: v.cpu() for k, v in H_dict.items()}, "h_dict.pt")
        log0("Full Hessian GPTQ Calibration complete.")'''

    source = source.replace(old_h_code, new_h_code)

    # In eval loading:
    old_eval_code = '''        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))'''

    new_eval_code = '''        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        import os
        if os.path.exists("h_dict.pt"):
            sd_cpu["_h_dict"] = torch.load("h_dict.pt", map_location="cpu", weights_only=False)
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))'''

    source = source.replace(old_eval_code, new_eval_code)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(source)

    print(f'Patched {filepath}')

process_file('train_gpt_phase3.py')
process_file('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')