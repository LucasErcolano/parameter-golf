import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    source = source.replace('H_inv = torch.linalg.inv(H + damp_val * torch.eye(H.size(0), device=H.device))',
                            'H_inv = torch.linalg.inv(H + (damp_val + 1e-4) * torch.eye(H.size(0), device=H.device))')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(source)

    print(f'Fixed damping in {filepath}')

if __name__ == "__main__":
    process_file('train_gpt_phase3.py')
    process_file('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')
