"""
Comprehensive audit of all Cooper lab files across cluster roots.

Usage (fast, no H5 inspection):
    python scripts/find_cooper_data.py

Usage (thorough, opens every H5 to check keys/shape/annotations):
    python scripts/find_cooper_data.py --check-h5 --out cooper_audit.csv

Custom roots / CSV:
    python scripts/find_cooper_data.py --roots /dir1 /dir2 --csv /path/meta.csv
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _md_table(series: pd.Series) -> str:
    """Render a Series as a simple markdown table without tabulate."""
    idx_hdr, val_hdr = series.index.name or "key", series.name or "value"
    rows = [(str(k), str(v)) for k, v in series.items()]
    w0 = max(len(idx_hdr), *(len(r[0]) for r in rows))
    w1 = max(len(val_hdr), *(len(r[1]) for r in rows))
    sep = f"| {'-'*w0} | {'-'*w1} |"
    hdr = f"| {idx_hdr:<{w0}} | {val_hdr:<{w1}} |"
    lines = [hdr, sep] + [f"| {r[0]:<{w0}} | {r[1]:<{w1}} |" for r in rows]
    return "\n".join(lines)

# ── defaults ────────────────────────────────────────────────────────────────
ROOTS_DEFAULT = [
    "/scratch-grete/projects/nim00007/data/mitochondria/cooper",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper",
]

# Files under these roots with *_combined.h5 names are in the cristae training set
CRISTAE_TRAINING_ROOTS = [
    "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae",
]

CSV_DEFAULT = Path(__file__).parent.parent / "cooper_data_overview.csv.bak"

# H5 keys we care about
KNOWN_KEYS = ["raw", "labels/cristae", "labels/mitochondria", "raw_mitos_combined", "seg"]


# ── stem normalisation ───────────────────────────────────────────────────────
_STRIP_RE = re.compile(r"(_combined|_s2|_downscaled)+$")


def normalize_stem(stem: str) -> str:
    """Iteratively strip processing suffixes to recover the CSV sample_stem."""
    prev = None
    while stem != prev:
        prev = stem
        stem = _STRIP_RE.sub("", stem)
    return stem


# ── training set membership ──────────────────────────────────────────────────
def is_cristae_training(path: Path) -> bool:
    if not path.name.endswith("_combined.h5"):
        return False
    for root in CRISTAE_TRAINING_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            pass
    return False


# ── H5 content inspection ────────────────────────────────────────────────────
def _top_keys(f: h5py.File) -> list:
    """List key datasets (KNOWN_KEYS present + any unknown top-level datasets)."""
    present = [k for k in KNOWN_KEYS if k in f]
    for tk in f.keys():
        if tk not in ("raw", "labels") and isinstance(f[tk], h5py.Dataset):
            fqk = tk
            if fqk not in present:
                present.append(fqk)
    return present


def h5_info(path: Path) -> dict:
    info = {
        "h5_keys": "",
        "raw_shape": "",
        "voxel_size_nm": "",
        "cristae_nonempty": "",
        "mito_nonempty": "",
        "combined_ready": False,
    }
    try:
        with h5py.File(path, "r") as f:
            present = _top_keys(f)
            info["h5_keys"] = ";".join(present)
            info["combined_ready"] = "raw_mitos_combined" in present

            if "raw" in f:
                ds = f["raw"]
                info["raw_shape"] = "×".join(str(d) for d in ds.shape) + f"({ds.dtype})"

            # Voxel size: file-level attrs first, then raw dataset attrs
            vs = f.attrs.get("voxel_size")
            if vs is None and "raw" in f:
                vs = f["raw"].attrs.get("voxel_size")
            if vs is not None:
                try:
                    arr = np.asarray(vs, dtype=float).flatten()
                    info["voxel_size_nm"] = ";".join(f"{v:.4f}" for v in arr)
                except Exception:
                    info["voxel_size_nm"] = str(vs)

            # Probe first 10 z-slices for non-empty annotations
            if "labels/cristae" in f:
                ds = f["labels/cristae"]
                nz = min(10, ds.shape[0])
                info["cristae_nonempty"] = bool(np.any(ds[:nz] > 0))

            for mk in ("labels/mitochondria", "labels/mito"):
                if mk in f:
                    ds = f[mk]
                    nz = min(10, ds.shape[0])
                    info["mito_nonempty"] = bool(np.any(ds[:nz] > 0))
                    break

    except Exception as exc:
        info["h5_keys"] = f"ERR:{exc}"
    return info


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Audit all Cooper lab files across cluster roots")
    parser.add_argument("--roots", nargs="+", default=ROOTS_DEFAULT,
                        help="Root directories to scan (default: both cluster roots)")
    parser.add_argument("--csv", default=str(CSV_DEFAULT),
                        help="Path to cooper_data_overview CSV (default: repo root .bak)")
    parser.add_argument("--out", default="cooper_data_audit.csv",
                        help="Output CSV path (default: cooper_data_audit.csv in cwd)")
    parser.add_argument("--check-h5", action="store_true",
                        help="Open each H5 and inspect datasets, shape, voxel size, annotations "
                             "(first-10-slice probe; slower but thorough)")
    args = parser.parse_args()

    # ── load CSV metadata ────────────────────────────────────────────────────
    csv_meta: dict = {}
    csv_path = Path(args.csv)
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path, sep=",", skipinitialspace=True)
        df_csv.columns = [c.strip() for c in df_csv.columns]
        for _, row in df_csv.iterrows():
            stem = str(row.get("sample_stem", "")).strip()
            if stem:
                csv_meta[stem] = {k: str(v).strip() for k, v in row.items()}
        print(f"Loaded {len(csv_meta)} rows from {csv_path.name}")
    else:
        print(f"WARNING: CSV not found at {args.csv}", file=sys.stderr)

    # ── discover files ───────────────────────────────────────────────────────
    rows = []
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            print(f"WARNING: root {root} does not exist — skipping", file=sys.stderr)
            continue
        print(f"\nScanning {root} …")

        # H5 and MRC: files
        for fmt_glob, fmt_name in [("*.h5", "h5"), ("*.mrc", "mrc"), ("*.rec", "rec")]:
            for path in sorted(root.rglob(fmt_glob)):
                if not path.is_file():
                    continue
                stem = path.stem
                norm = normalize_stem(stem)
                meta = csv_meta.get(norm, {})
                csv_match = "YES" if meta else "NO"

                row = {
                    "path": str(path),
                    "root": root_str,
                    "format": fmt_name,
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                    "stem": stem,
                    "norm_stem": norm,
                    "csv_match": csv_match,
                    "specimen_id": meta.get("specimen_id", ""),
                    "cell_line": meta.get("cell_line", ""),
                    "genotype": meta.get("genotype", ""),
                    "condition": meta.get("condition", ""),
                    "session_date": meta.get("session_date", ""),
                    "experiment": meta.get("experiment", ""),
                    "csv_has_mito": meta.get("has_mito_labels", ""),
                    "csv_has_cristae": meta.get("has_cristae_labels", ""),
                    "h5_keys": "",
                    "raw_shape": "",
                    "voxel_size_nm": "",
                    "cristae_nonempty": "",
                    "mito_nonempty": "",
                    "combined_ready": "",
                    "cristae_training": is_cristae_training(path),
                }

                if args.check_h5 and fmt_name == "h5":
                    row.update(h5_info(path))

                rows.append(row)
                tag = f"[{fmt_name:3s}][csv:{csv_match}][cristae_train:{str(row['cristae_training'])[0]}]"
                print(f"  {tag} {path.name[:72]}")

        # Zarr / N5: directories with matching extension
        for dir_glob in ("*.zarr", "*.n5"):
            for path in sorted(root.rglob(dir_glob)):
                if not path.is_dir():
                    continue
                fmt_name = path.suffix.lstrip(".")
                size_mb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)
                stem = path.stem
                norm = normalize_stem(stem)
                meta = csv_meta.get(norm, {})
                csv_match = "YES" if meta else "NO"
                row = {
                    "path": str(path),
                    "root": root_str,
                    "format": fmt_name,
                    "size_mb": size_mb,
                    "stem": stem,
                    "norm_stem": norm,
                    "csv_match": csv_match,
                    "specimen_id": meta.get("specimen_id", ""),
                    "cell_line": meta.get("cell_line", ""),
                    "genotype": meta.get("genotype", ""),
                    "condition": meta.get("condition", ""),
                    "session_date": meta.get("session_date", ""),
                    "experiment": meta.get("experiment", ""),
                    "csv_has_mito": meta.get("has_mito_labels", ""),
                    "csv_has_cristae": meta.get("has_cristae_labels", ""),
                    "h5_keys": "", "raw_shape": "", "voxel_size_nm": "",
                    "cristae_nonempty": "", "mito_nonempty": "", "combined_ready": "",
                    "cristae_training": False,
                }
                rows.append(row)
                tag = f"[{fmt_name:4s}][csv:{csv_match}]"
                print(f"  {tag} {path.name[:72]}")

    # ── write CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows → {args.out}")

    # ── markdown summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("## Cooper data audit summary")
    print("=" * 60)

    print("\n### Files by format")
    fmt_counts = df.groupby("format").size().rename("count")
    print(_md_table(fmt_counts))

    print("\n### CSV match rate (across all formats)")
    print(_md_table(df["csv_match"].value_counts().rename("count")))

    unmatched = df[df["csv_match"] == "NO"][["format", "stem"]].drop_duplicates()
    if len(unmatched):
        print(f"\n  Unmatched stems (first 20):")
        for _, r in unmatched.head(20).iterrows():
            print(f"    [{r['format']}] {r['stem']}")

    print("\n### Cristae training set")
    ct = df[df["cristae_training"] == True]
    print(f"  {len(ct)} files in cristae training set")
    if len(ct):
        for _, r in ct.iterrows():
            print(f"    {r['stem']}")

    if args.check_h5:
        h5_df = df[df["format"] == "h5"]
        cris_yes = (h5_df["cristae_nonempty"] == True).sum()
        mito_yes  = (h5_df["mito_nonempty"]  == True).sum()
        comb_yes  = (h5_df["combined_ready"] == True).sum()
        print(f"\n### H5 annotation probe (first-10-slice)")
        print(f"  labels/cristae non-empty : {cris_yes} / {len(h5_df)}")
        print(f"  labels/mito    non-empty : {mito_yes} / {len(h5_df)}")
        print(f"  raw_mitos_combined ready : {comb_yes} / {len(h5_df)}")

        # Files with cristae but NOT in cristae training set
        missed = h5_df[(h5_df["cristae_nonempty"] == True) & (h5_df["cristae_training"] == False)]
        if len(missed):
            print(f"\n  H5 files with cristae annotations NOT in cristae training ({len(missed)}):")
            for _, r in missed.iterrows():
                print(f"    {r['path']}")

    gt = df[df["csv_match"] == "YES"]["genotype"].value_counts()
    if len(gt):
        print("\n### Genotype distribution (CSV-matched files)")
        print(_md_table(gt.rename("files")))


if __name__ == "__main__":
    main()
