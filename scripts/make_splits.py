"""Create stratified train/val/test split lists (CSV) from the audit table.

- drops unreadable files and exact duplicates (and hashes that appear in BOTH classes)
- converts formats TensorFlow cannot decode (e.g. WEBP, MPO, CMYK JPEG, 16-bit) to RGB PNG in data/converted
- label: 1 = AI-generated, 0 = Real
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None
TF_SAFE = {("JPEG", "RGB"), ("JPEG", "L"), ("PNG", "RGB"), ("PNG", "RGBA"), ("PNG", "L"),
           ("PNG", "P"), ("PNG", "LA"), ("BMP", "RGB"), ("GIF", "P"), ("GIF", "L")}


def label_of(class_name: str) -> int:
    n = class_name.lower()
    if n.startswith("ai") or "fake" in n or "generated" in n:
        return 1
    if "real" in n:
        return 0
    raise ValueError(f"Cannot infer label from folder name {class_name!r}")


def main(audit, out, conv_dir, seed, val, test):
    df = pd.read_csv(audit)
    df = df[df.readable]
    conflict = df.groupby("sha256")["class"].nunique()
    df = df[~df.sha256.isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("sha256", keep="first").copy()
    df["label"] = df["class"].map(label_of)

    conv_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for r in df.itertuples():
        if (r.format, r.mode) in TF_SAFE:
            new_paths.append(r.path)
            continue
        dst = conv_dir / f"{r.sha256[:16]}.png"
        if not dst.exists():
            with Image.open(r.path) as im:
                im.convert("RGB").save(dst)
        new_paths.append(dst.as_posix())
    df["path"] = new_paths
    print(f"Converted {sum(p.startswith(conv_dir.as_posix()) for p in new_paths)} files to PNG")

    cols = ["path", "label", "class", "sha256"]
    train, rest = train_test_split(df[cols], test_size=val + test, stratify=df.label, random_state=seed)
    vdf, tdf = train_test_split(rest, test_size=test / (val + test), stratify=rest.label, random_state=seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", vdf), ("test", tdf)]:
        part.sort_values("path").to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s} {len(part):5d}  AI={int(part.label.sum())}  Real={int((part.label == 0).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/audit.csv")
    ap.add_argument("--out", default="data/splits", type=Path)
    ap.add_argument("--converted", default="data/converted", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    a = ap.parse_args()
    main(a.audit, a.out, a.converted, a.seed, a.val, a.test)
