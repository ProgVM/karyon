import json
import subprocess
import os
import sys

def main():
    # 1. Compile CUDA source
    compile_cmd = "nvcc -O3 -arch=sm_75 -use_fast_math -Xptxas -O3 experiments/exp_734_pure_cuda_multiverse_autopoiesis.cu -o bin_exp_734"
    res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation Failed:\n{res.stderr}")
        sys.exit(1)
        
    # 2. Run pure native binary
    run_res = subprocess.run("./bin_exp_734", shell=True, capture_output=True, text=True)
    print(run_res.stdout)
    if run_res.returncode != 0:
        print(f"Execution Failed:\n{run_res.stderr}")
        sys.exit(1)
        
    # 3. Read json
    if os.path.exists("experiments/exp_734_results.json"):
        with open("experiments/exp_734_results.json", "r") as f:
            data = json.load(f)
        print(f"Final Chunk Loss : {data.get('loss')}")
        print(f"PPL              : {data.get('ppl')}")
        print(f"Delta Loss       : {data.get('delta_loss')}")
        print(f"Verdict          : {data.get('verdict')}")

if __name__ == "__main__":
    main()
