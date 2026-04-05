import subprocess
import os

def run_test(run_id, seed, steps=10):
    env = os.environ.copy()
    env["TORCH_COMPILE"] = "0"
    env["SEED"] = str(seed)
    env["ITERATIONS"] = "5000" # So it doesn't hit last_step and eval
    
    proc = subprocess.Popen(
        [".venv311\\Scripts\\python.exe", "train_gpt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    losses = []
    for line in proc.stdout:
        print(line, end="")
        if "train_loss:" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("train_loss:"):
                    losses.append(p.split(":")[1])
                    break
        if len(losses) >= steps:
            proc.terminate()
            proc.wait()
            break
    return losses

losses1 = run_test("run1", 42)
print("--- RUN 1 DONE ---")
losses2 = run_test("run2", 42)
print("--- RUN 2 DONE ---")

print("Run 1 losses:", losses1)
print("Run 2 losses:", losses2)

if losses1 == losses2:
    print("DETERMINISM TEST PASSED")
else:
    print("DETERMINISM TEST FAILED")
