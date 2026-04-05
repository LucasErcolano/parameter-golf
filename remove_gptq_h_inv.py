import re

def process(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove def gptq_quantize_tensor
    content = re.sub(r'def gptq_quantize_tensor\(.*?\n\ndef mixed_quantize_int6', 'def mixed_quantize_int6', content, flags=re.DOTALL)

    # Revert mixed_quantize_int6: _h_dict references
    content = content.replace('''    for name, tensor in state_dict.items():
        if name == "_h_dict":
            continue
        t = tensor.detach().cpu().contiguous()''', '''    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()''')

    # Revert H_dict logic in mixed_quantize_int6
    old_mixed = '''        if bits in (5, 6) and t.ndim >= 1:
            H_dict = state_dict.get("_h_dict", {})
            module_name = name.rsplit('.', 1)[0] if '.' in name else name
            if module_name in H_dict:
                q, s = gptq_quantize_tensor(t, H_dict[module_name], bits)
            else:
                q, s = quantize_signed_per_row(t, bits)
            q_packed, q_numel = pack_signed_tensor(q, bits)'''
    new_mixed = '''        if bits in (5, 6) and t.ndim >= 1:
            q, s = quantize_signed_per_row(t, bits)
            q_packed, q_numel = pack_signed_tensor(q, bits)'''
    content = content.replace(old_mixed, new_mixed)

    # Remove H_dict extraction in eval block
    old_eval_dict = '''        full_state_dict = base_model.state_dict()
        export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        if os.path.exists("h_dict.pt"):
            sd_cpu["_h_dict"] = torch.load("h_dict.pt", map_location="cpu", weights_only=False)
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))'''
    new_eval_dict = '''        full_state_dict = base_model.state_dict()
        export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
        sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, get_quant_bits_by_cat(args))'''
    content = content.replace(old_eval_dict, new_eval_dict)

    # Remove the Full Hessian GPTQ Calibration block from main
    content = re.sub(r'    # --- Full Hessian GPTQ Calibration ---.*?    # ------------------------------------\n', '', content, flags=re.DOTALL)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done", filepath)

process('train_gpt_phase3.py')
process('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')