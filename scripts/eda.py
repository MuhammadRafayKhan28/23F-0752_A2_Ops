"""Task 1.3 - visual EDA: sample grid per class + histograms of width, height, file size."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def sample_grid(df, out, n=8, seed=42):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), n, figsize=(2.2 * n, 2.4 * len(classes)))
    for i, c in enumerate(classes):
        sub = df[df["class"] == c].sample(min(n, (df["class"] == c).sum()), random_state=seed)
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(sub):
                r = sub.iloc[j]
                with Image.open(r.path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((256, 256))
                    ax.imshow(im)
                ax.set_title(f"{c}\n{r.width}x{r.height}", fontsize=8)
    fig.suptitle("Random samples per class")
    fig.tight_layout()
    fig.savefig(out / "sample_grid.png", dpi=120)
    plt.close(fig)


def histograms(df, out):
    df = df.assign(file_size_kb=df.file_size / 1024)
    for col, label, log in [("width", "Width (px)", False), ("height", "Height (px)", False),
                            ("file_size_kb", "File size (KB)", True)]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, g in df.groupby("class"):
            ax.hist(g[col], bins=40, alpha=0.55, label=f"{c} (median {g[col].median():.0f})")
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Images")
        ax.set_title(f"{label} per class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"hist_{col}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/audit.csv")
    ap.add_argument("--out", default="reports/figures", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.audit)
    data = data[data.readable & ~data.is_duplicate]
    sample_grid(data, a.out)
    histograms(data, a.out)
    print(f"Figures written to {a.out}")
