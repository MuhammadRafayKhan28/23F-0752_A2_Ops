"""Task 3.5 - launch the experiment grid. Each run changes ONE thing vs. the reference run R2.

python scripts/run_experiments.py            # all
python scripts/run_experiments.py R1 R2      # subset
"""
import subprocess
import sys

REF = ["--model", "mobilenetv2", "--lr", "1e-3", "--img-size", "224"]
RUNS = {
    "R1": ("cnn_baseline", ["--model", "cnn", "--lr", "1e-3", "--epochs", "40", "--patience", "8"]),
    "R2": ("mnv2_ref", REF),                                                   # reference
    "R3": ("mnv2_lr3e-4", REF[:2] + ["--lr", "3e-4", "--img-size", "224"]),    # learning rate
    "R4": ("mnv2_noaug", REF + ["--no-aug"]),                                  # augmentation
    "R5": ("mnv2_frozen_only", REF + ["--finetune", "none"]),                  # frozen layers
    "R6": ("mnv2_img160", REF[:4] + ["--img-size", "160"]),                    # image size
    "R7": ("effb0_ref", ["--model", "efficientnetb0", "--lr", "1e-3"]),        # backbone
    "R8": ("mnv2_ref_autolog", REF + ["--autolog"]),                           # autolog approach (3.3)
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(RUNS)
    for key in chosen:
        name, args = RUNS[key]
        cmd = [sys.executable, "-m", "src.train", "--run-name", name, *args]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
