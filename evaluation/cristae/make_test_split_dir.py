"""Stage the pinned cristae test split into one directory of symlinks.

The synapse-net `run_segmentation` CLI iterates a directory of inputs; our 15 test files
(SYNAPSENETV1_TEST_SPLIT in training/cristae/train_cristae.py) live in scattered source dirs.
This links them into a single dir so the CLI (and evaluate_cristae.py) can consume them.
The combined h5 (`raw_mitos_combined` [2,Z,Y,X] + `labels/cristae`) serves as BOTH the CLI input
and the GT for evaluation.
"""
import argparse
import os
import re


def load_test_split(train_script):
    txt = open(train_script).read()
    m = re.search(r"SYNAPSENETV1_TEST_SPLIT\s*=\s*\[(.*?)\]", txt, re.S)
    if m is None:
        raise RuntimeError(f"Could not find SYNAPSENETV1_TEST_SPLIT in {train_script}")
    return re.findall(r"[\"']([^\"']+\.h5)[\"']", m.group(1))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_train = os.path.normpath(os.path.join(here, "..", "..", "training", "cristae", "train_cristae.py"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Directory to populate with symlinks.")
    ap.add_argument("--train_script", default=default_train)
    ap.add_argument("--exclude", action="append", default=[],
                    help="Substring(s) of basenames to skip (repeatable). E.g. a file whose "
                         "raw_mitos_combined is float32 and crashes synapse-net's regionprops.")
    args = ap.parse_args()

    files = load_test_split(args.train_script)
    os.makedirs(args.out, exist_ok=True)

    # Remove any stale symlinks so an --exclude'd file does not linger from a previous run.
    for old in os.listdir(args.out):
        p = os.path.join(args.out, old)
        if os.path.islink(p):
            os.remove(p)

    seen = {}
    n = 0
    skipped = 0
    for f in files:
        if any(sub in os.path.basename(f) for sub in args.exclude):
            print(f"  excluded: {os.path.basename(f)}")
            skipped += 1
            continue
        if not os.path.exists(f):
            raise FileNotFoundError(f"test file missing: {f}")
        base = os.path.basename(f)
        if base in seen:
            raise RuntimeError(f"basename collision: {base}\n  {seen[base]}\n  {f}")
        seen[base] = f
        dst = os.path.join(args.out, base)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(f, dst)
        n += 1
    print(f"linked {n} test files into {args.out}")


if __name__ == "__main__":
    main()
