"""Leakage-safe, stratified train/val splitting for cristae training.

Background: the previous split (`synapse.util.split_data_paths_to_dict_with_ensure`)
did a flat positional cut after a random shuffle, with `ensure_strings` only
guaranteeing one file per substring. That (a) let sibling crops of the *same*
specimen land in both train and val (leaky, over-optimistic validation) and
(b) under-represented rare groups (KO, Otof-KO, cooper) in val. A validation set
with those flaws cannot reveal a regression on hard/rare files (see
`evaluation/cristae/DIAGNOSIS_cristae_06-01_vs_06-04.md`).

This module groups files by **specimen** (so whole specimens go to one split) and
**stratifies** the validation set across (source, genotype) so every group with
enough specimens is represented in val. Pure-python (no torch/numpy) so it can be
audited on a login node via the `__main__` dry-run.
"""
import os
import re
from collections import defaultdict

# Filename taxonomy. Order matters: Otof first, then cooper tomogram ids, then WT/KO/M animals.
_OTOF_RE = re.compile(r"^(Otof_AVCN\d+_[0-9A-Za-z]+)_(WT|KO)")
_COOPER_RE = re.compile(r"^(\d+_[A-Za-z]\d+)")
_ANIMAL_RE = re.compile(r"^(WT|KO|M)(\d+)")


def parse_specimen(path):
    """Return (source, genotype, specimen) parsed from a file path.

    - source:   "cooper" if the path is under a cooper root, else "wichmann".
    - genotype: "Otof-WT"/"Otof-KO", "cooper", or the animal line "WT"/"KO"/"M";
                "other" if nothing matches (surfaced by the dry-run audit).
    - specimen: the grouping unit — all crops of one specimen share it
                (e.g. "Otof_AVCN07_455L", "36194_B4", "WT20").
    """
    base = os.path.basename(path)
    source = "cooper" if "cooper" in path else "wichmann"

    m = _OTOF_RE.match(base)
    if m:
        return source, f"Otof-{m.group(2)}", m.group(1)
    m = _COOPER_RE.match(base)
    if m:
        return source, "cooper", m.group(1)
    m = _ANIMAL_RE.match(base)
    if m:
        return source, m.group(1), f"{m.group(1)}{m.group(2)}"
    return source, "other", base.split("_")[0]


def grouped_stratified_split(
    data_paths,
    val_ratio=0.1,
    seed=42,
    pinned_test=None,
    holdout_test_siblings=False,
    verbose=True,
):
    """Split `data_paths` into {train, val, test}, grouping by specimen.

    - Whole specimens are assigned to a single split (no sibling-crop leakage).
    - Validation is stratified: every (source, genotype) stratum that has >= 2
      specimens contributes >= 1 specimen to val, targeting ~`val_ratio` of that
      stratum's files, while always keeping its largest specimen in train (so a
      represented stratum is never left with an empty train side). Singleton strata
      stay in train (reported as unvalidated).
    - `pinned_test`: an explicit list of test files. When `holdout_test_siblings`
      is True, *all* sibling crops of the test specimens are removed from the
      train/val pool so the test set is genuinely specimen-disjoint.

    Returns a dict with "train"/"val"/"test" (sorted lists of paths), same shape
    as `split_data_paths_to_dict_with_ensure`.
    """
    _ = seed  # ordering is deterministic by specimen size (smallest-first); seed kept for API stability
    pinned_test = list(pinned_test) if pinned_test else []
    pinned_set = set(pinned_test)
    pool = [p for p in data_paths if p not in pinned_set]

    test_specs = {(parse_specimen(p)[0], parse_specimen(p)[2]) for p in pinned_test}

    n_dropped = 0
    if holdout_test_siblings and pinned_test:
        kept = []
        for p in pool:
            src, _, spec = parse_specimen(p)
            if (src, spec) in test_specs:
                n_dropped += 1
            else:
                kept.append(p)
        pool = kept

    # group pool files by (source, specimen); record each group's stratum
    groups = defaultdict(list)
    group_stratum = {}
    for p in pool:
        src, geno, spec = parse_specimen(p)
        key = (src, spec)
        groups[key].append(p)
        group_stratum[key] = (src, geno)

    # bucket specimens by stratum
    stratum_specs = defaultdict(list)
    for key, stratum in group_stratum.items():
        stratum_specs[stratum].append(key)

    val_files, train_files, unvalidated = [], [], []
    for stratum in sorted(stratum_specs):
        # smallest specimens first: gives val stratum-coverage while keeping val near
        # the target size and leaving the large (data-rich) specimens for training.
        keys = sorted(stratum_specs[stratum], key=lambda k: (len(groups[k]), k))
        n_files = sum(len(groups[k]) for k in keys)
        if len(keys) < 2:
            unvalidated.append(stratum)
            target_val = 0
        else:
            target_val = max(1, round(val_ratio * n_files))
        v = 0
        n_keys = len(keys)
        for i, k in enumerate(keys):
            # keep the largest specimen (last, ascending sort) in train so a
            # multi-specimen stratum is never left with an empty train side.
            is_last = i == n_keys - 1
            if v < target_val and not is_last:
                val_files.extend(groups[k])
                v += len(groups[k])
            else:
                train_files.extend(groups[k])

    if verbose:
        print(f"[grouped_stratified_split] pool={len(pool)} files "
              f"(dropped {n_dropped} test-sibling crops), "
              f"train={len(train_files)} val={len(val_files)} test={len(pinned_test)}")
        if unvalidated:
            print(f"[grouped_stratified_split] strata with <2 specimens (train-only, unvalidated): "
                  f"{sorted(unvalidated)}")

    return {"train": sorted(train_files), "val": sorted(val_files), "test": sorted(pinned_test)}


def summarize_split(split, strict_test=True):
    """Print a per-stratum (files | specimens) table and assert no leakage.

    Always asserts train and val are specimen-disjoint. Test overlap with
    train/val raises when `strict_test` (use with holdout_test_siblings=True),
    otherwise it is only warned about.
    """
    rows = {}
    spec_splits = defaultdict(set)
    for name in ("train", "val", "test"):
        for p in split.get(name, []):
            src, geno, spec = parse_specimen(p)
            stratum = (src, geno)
            r = rows.setdefault(stratum, {"train": [0, set()], "val": [0, set()], "test": [0, set()]})
            r[name][0] += 1
            r[name][1].add((src, spec))
            spec_splits[(src, spec)].add(name)

    print("\n=== split composition: files | specimens (per stratum) ===")
    for stratum in sorted(rows):
        r = rows[stratum]
        print(f"  {str(stratum):26} "
              f"train {r['train'][0]:4}|{len(r['train'][1]):2}   "
              f"val {r['val'][0]:3}|{len(r['val'][1]):2}   "
              f"test {r['test'][0]:3}|{len(r['test'][1]):2}")

    # leakage checks
    train_val_leak = {k: v for k, v in spec_splits.items() if {"train", "val"} <= v}
    if train_val_leak:
        raise AssertionError(f"Specimen leakage between train and val: {sorted(train_val_leak)}")
    test_leak = {k: v for k, v in spec_splits.items() if "test" in v and (v - {"test"})}
    if test_leak:
        msg = f"Test specimens also in train/val: {sorted(test_leak)}"
        if strict_test:
            raise AssertionError(msg)
        print(f"  [WARN] {msg}")
    print("  [leakage check OK: train/val specimen-disjoint"
          + ("" if strict_test else "; test overlap allowed") + "]")

    val_strata = {s for s in rows if rows[s]["val"][0] > 0}
    missing = sorted(s for s in rows if s not in val_strata)
    if missing:
        print(f"  [strata with NO val coverage: {missing}]")

    no_train = sorted(s for s in rows if rows[s]["train"][0] == 0 and rows[s]["val"][0] > 0)
    if no_train:
        print(f"  [strata with NO train coverage: {no_train}]")


# Default data roots (mirror training/cristae/train_cristae.py) for the dry-run audit.
_DEFAULT_ROOTS = [
    "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/",
]
_EXCLUDE = [
    "Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined",
    "WT20_eb8_AZ1_model_combined",
    "WT22_eb8_model_combined",
]


def _dry_run():
    """Audit the split over the real data roots (paths only, no H5 reads)."""
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Dry-run audit of the grouped/stratified cristae split.")
    ap.add_argument("--roots", nargs="+", default=_DEFAULT_ROOTS)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout_test_siblings", action="store_true",
                    help="Opt in to leakage-safe test (drops test-specimen sibling crops from train/val; "
                         "costly in data). Off by default, matching grouped_stratified_split.")
    args = ap.parse_args()

    paths = []
    for root in args.roots:
        paths.extend(glob.glob(os.path.join(root, "**", "*_combined.h5"), recursive=True))
    paths = sorted(set(p for p in paths if "_combined.h5" in p))
    for s in _EXCLUDE:
        paths = [p for p in paths if s not in p]
    print(f"found {len(paths)} files across {len(args.roots)} roots")

    # surface parse-fallbacks
    others = [p for p in paths if parse_specimen(p)[1] == "other"]
    if others:
        print(f"[WARN] {len(others)} files fell into the 'other' genotype bucket (parse gaps):")
        for p in others[:20]:
            print("   ", os.path.basename(p))

    pinned = None
    try:  # best-effort: reuse the canonical pinned test split if importable
        from training.cristae.train_cristae import SYNAPSENETV1_TEST_SPLIT as pinned  # noqa
    except Exception as e:
        print(f"[dry-run] could not import pinned test split ({e}); auditing train/val only")

    split = grouped_stratified_split(
        paths, val_ratio=args.val_ratio, seed=args.seed,
        pinned_test=pinned, holdout_test_siblings=args.holdout_test_siblings,
    )
    summarize_split(split, strict_test=args.holdout_test_siblings)


if __name__ == "__main__":
    _dry_run()
