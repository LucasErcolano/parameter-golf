import ast
import re
import sys

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # Update XSA_LAST_N
    source = re.sub(
        r'xsa_last_n = int\(os\.environ\.get\("XSA_LAST_N", \d+\)\)',
        'xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))',
        source
    )

    class CodeRemover(ast.NodeVisitor):
        def __init__(self):
            self.to_remove = []
            
        def visit_ClassDef(self, node):
            if node.name in ['BackoffNgramMixer', 'TrainNgramTracker', 'TTTLoRAAdapter', 'LoRAAdapter', 'HashedNgramMixer']:
                self.to_remove.append((node.lineno, node.end_lineno))
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            if node.name in ['build_ttt_chunk_windows', 'build_ttt_optimizer', 'get_even_ttt_seq_span', 'eval_val_sliding_ttt', 'train_step', 'train_lora_on_chunk']:
                self.to_remove.append((node.lineno, node.end_lineno))
            self.generic_visit(node)

    tree = ast.parse(source)
    remover = CodeRemover()
    remover.visit(tree)

    lines = source.splitlines()
    to_delete = set()
    for start, end in remover.to_remove:
        for i in range(start-1, end):
            to_delete.add(i)

    new_lines = [line for i, line in enumerate(lines) if i not in to_delete]
    new_source = '\n'.join(new_lines) + '\n'

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_source)

    print(f'Processed {filepath}')

if __name__ == "__main__":
    process_file('train_gpt_phase3.py')
    process_file('records/track_10min_16mb/2026-04-04_LucasErcolano_MixedQuantNgram/train_gpt.py')
