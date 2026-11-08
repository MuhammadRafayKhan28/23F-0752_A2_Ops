"""Task 1.2 - dataset audit.

Walks data/raw/<class_folder>/... and builds one row per file with:
class, path, file size, width, height, colour mode, format, SHA-256, readable flag.
Writes reports/audit.csv and reports/audit_summary.md
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some real photos are very large; we only read headers + verify


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"width": None, "height": None, "mode": None, "format": None, "readable": True, "error": ""}
    try:
        with Image.open(path) as im:          # 1) header parse
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.verify()                        # 2) structural check
        with Image.open(path) as im:          # verify() invalidates the object -> reopen
            im.load()                          # 3) full pixel decode catches truncated files
    except Exception as e:  # noqa: BLE001
        row.update(readable=False, error=f"{type(e).__name__}: {e}"[:200])
    return row


def main(root: Path, out: Path):
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {root}")
    rows = []
    for cdir in class_dirs:
        for p in sorted(cdir.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            r = {"class": cdir.name, "path": p.as_posix(), "file_size": p.stat().st_size,
                 "extension": p.suffix.lower(), "sha256": sha256_of(p)}
            r.update(inspect(p))
            rows.append(r)
    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "audit.csv", index=False)

    ok = df[df.readable]
    counts = df.groupby("class").agg(files=("path", "size"), readable=("readable", "sum"),
                                     corrupt=("readable", lambda s: int((~s).sum())),
                                     duplicates=("is_duplicate", "sum"))
    stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    dup_groups = df[df.sha256.duplicated(keep=False)].groupby("sha256")["class"].agg(lambda s: sorted(set(s)))
    cross = int(sum(len(c) > 1 for c in dup_groups))
    md = [
        "# Dataset audit summary\n",
        "## Counts per class\n", counts.to_markdown(), "\n",
        "## Width / height / file size (readable images)\n", stats.to_markdown(), "\n",
        "## Formats and colour modes\n",
        pd.crosstab(ok["class"], ok["format"]).to_markdown(), "\n",
        pd.crosstab(ok["class"], ok["mode"]).to_markdown(), "\n",
        f"- Total files: {len(df)}",
        f"- Corrupt / unreadable: {int((~df.readable).sum())}",
        f"- Exact duplicate files (extra copies): {int(df.is_duplicate.sum())}",
        f"- Duplicate groups spanning BOTH classes (label conflicts): {cross}",
    ]
    if (~df.readable).any():
        md += ["\n## Unreadable files\n", df.loc[~df.readable, ["path", "error"]].to_markdown(index=False)]
    (out / "audit_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw", type=Path)
    ap.add_argument("--out", default="reports", type=Path)
    a = ap.parse_args()
    main(a.root, a.out)
