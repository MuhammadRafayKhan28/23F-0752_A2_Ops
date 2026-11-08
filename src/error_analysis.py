"""Task 2.6 - grid of misclassified test images with an automatic first hypothesis for each.

python -m src.error_analysis --run outputs/mobilenetv2_lr0.001_224
Edit the generated reports/error_analysis.md by hand after looking at the images.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def image_stats(path):
    with Image.open(path) as im:
        w, h = im.size
        rgb = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.float32)
    g = rgb.mean(axis=2)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    sat = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    return {"w": w, "h": h, "sharpness": float(lap.var()), "saturation": float(sat)}


def hypothesis(row, s):
    true = "Real" if row.label == 0 else "AI"
    hints = []
    if s["sharpness"] < 60:
        hints.append("very smooth/blurred - lost the fine texture cues")
    if s["sharpness"] > 1500:
        hints.append("heavy high-frequency texture/noise (detailed or over-sharpened)")
    if s["saturation"] > 90:
        hints.append("hyper-saturated colours look 'rendered'")
    if s["saturation"] < 15:
        hints.append("near-greyscale; colour statistics uninformative")
    if min(s["w"], s["h"]) < 300:
        hints.append("small source image, upscaling to 224 adds blur")
    if max(s["w"], s["h"]) / max(1, min(s["w"], s["h"])) > 1.8:
        hints.append("extreme aspect ratio distorted by stretching to a square")
    if not hints:
        hints.append("painting/illustration style or studio-like lighting resembles the other class"
                     if true == "Real" else "photorealistic generator output with natural lighting")
    return "; ".join(hints)


def main(run_dir: Path, n: int, out: Path):
    df = pd.read_csv(run_dir / "test_predictions.csv")
    wrong = df[df.label != df.pred].copy()
    wrong["err"] = (wrong.prob_ai - wrong.label).abs()
    wrong = wrong.sort_values("err", ascending=False).head(n)
    cols = 5
    rows = max(1, (len(wrong) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    md = ["# Error analysis - most confident mistakes on the test set\n",
          "| # | file | true | predicted P(AI) | size | sharpness | saturation | hypothesis |",
          "|---|---|---|---|---|---|---|---|"]
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if i >= len(wrong):
            continue
        r = wrong.iloc[i]
        s = image_stats(r.path)
        with Image.open(r.path) as im:
            ax.imshow(im.convert("RGB"))
        true = "Real" if r.label == 0 else "AI"
        ax.set_title(f"#{i + 1} true={true}  P(AI)={r.prob_ai:.2f}", fontsize=9)
        md.append(f"| {i + 1} | {Path(r.path).name} | {true} | {r.prob_ai:.2f} | {s['w']}x{s['h']} | "
                  f"{s['sharpness']:.0f} | {s['saturation']:.0f} | {hypothesis(r, s)} |")
    fig.suptitle("Misclassified test images")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "misclassified_grid.png", dpi=110)
    (out / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    a = ap.parse_args()
    main(a.run, a.n, a.out)
