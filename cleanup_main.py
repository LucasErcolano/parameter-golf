import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # Remove eval_val_sliding_ttt calls
    source = re.sub(r'if args\.ttt_enabled or args\.use_hedge_mixer:.*?eval:(ttt_lora|ngram).*?time\.perf_counter\(\)-t_ttt\):\.0f}ms"\)', '', source, flags=re.DOTALL)
    
    source = re.sub(r'if args\.ttt_enabled or args\.use_hedge_mixer:.*?metric_name = "ttt".*?log0\(f"{metric_name}_x vl:{ttt_val_loss:\.8f} bpb:{ttt_val_bpb:\.8f}"\)', '', source, flags=re.DOTALL)

    # Remove references to Base model's ngram tracker if they exist
    source = re.sub(r'base_model\._ngram_tracker = .*?\n', '', source)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(source)

    print(f'Cleaned up {filepath}')

if __name__ == "__main__":
    process_file('train_gpt_phase3.py')
    process_file('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')
