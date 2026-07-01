# Cristae Training Data — Inventory & Notes

_Snapshot: 2026-06-24. Source of truth for discovery: `configs/training/cristae/cristae_persample_aug.yaml`
and `training/cristae/train_cristae.py`._

## Where the data lives (3 discovery roots)

Training discovers every `*_combined.h5` across these three dirs, pins a fixed 15-file test set, and
splits the rest into train/val:

| role | path | files |
|---|---|---|
| `data_dir`  | `/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2` | 4 (cooper, s2-binned) |
| `data_dir2` | `/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae` | 18 (cooper; incl. 5 new in `for_Luca_2026/`) |
| `data_dir3` | `/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/` | 128 (wichmann) |

- ≈**150** cristae-labeled candidates total (+5 new cooper 2026-06-24). After pinning 15 test and
  dropping ~2 (fail `min_shape`), the last run (`persample-aug`, 2026-06-22) used **train 116 / val 12 /
  test 15 = 143** (on the 145-file set; next run will have ~148 train/val candidates).
- Every file has keys `raw_mitos_combined` + `labels/cristae` (+ `labels/mitochondria`).
- **Input is 2-channel** (raw EM + mitochondria mask); model is 2-in / 2-out.
- Poor-quality files moved out on 2026-06-17 live in the sibling dir
  `/mnt/lustre-grete/usr/u12103/cristae_data/wichmann_needs_corrections/` (outside `data_dir3` →
  automatically excluded from discovery).

## Contribution by source — wichmann dominates (~88%)

| source | files | share | subsets |
|---|---|---|---|
| **wichmann** | 128 | 88% | `inital_data` 115 (eb/syn) + `2026-05-26` Otof 13 |
| **cooper**   | 17  | 12% | `cristae` 13 + `raw_mito_combined_s2` 4 |

## Genotype (wild-type vs knockout)

**Wichmann (128):** WT **77** · KO **22** · unspecified **29** (the `M#_eb*` mouse lines — genotype not
encoded in the filename). Otof subset alone: KO 13 / WT 10.

**Cooper (22):** `36194_B4` (5) and `36859_J1/J2` (J1 7, J2 4) are all **WT** (per
`cooper_data_overview.csv.bak`). The **37371** experiment is a clean DKO-vs-CTRL pair — confirmed by the
cooper lab tomo listing below (matches the CSV genotype column exactly):

- **DKO (M13DKO) — `37371_O4`, 5 tomos:** `SC_50`, `SC_52`, `SC_54`, `SC_57`, `SC_59`
- **CTRL (WT) — `37371_O5`, 5 tomos:** `SP_34-01`, `SP_35`, `SP_38-01`, `SP_67`, `mod_…SP_34-02`

Only **3 of those 10** carry cristae labels and reach the corpus: **1 DKO** (`O4_SC_52`, data_dir2) +
**2 CTRL** (`O5_SP_34-01` data_dir2; `O5_SP_35` in `raw_mito_combined_s2`). `O5_SP_35`'s `labels/cristae`
was verified non-empty on 2026-06-25 (305,713 voxels, instance-labeled) — the CSV's blank `cristae` field
for it is incomplete, not correct. No unknowns in cooper.

Net labeled balance ≈ **WT/CTRL ~98 vs KO/DKO ~23**, with 29 wichmann `M#_eb*` files carrying no
genotype label (mouse number only). Coverage is strongly WT-skewed (~81% WT among labeled files).

## Voxel size — uniform ~1.74 nm isotropic (verified from MRC headers)

| source | original (MRC) | as used in training | how verified |
|---|---|---|---|
| wichmann Otof | 1.748 nm | 1.748 nm (native; H5 dims == MRC dims) | 26 MRC originals in `cristae_data/wichmann/2026-05-26_original_files/2026-05-26/` |
| cooper cristae | 0.869 nm | ~1.738 nm (**2× binned**) | cooper MRC headers + H5 attrs |
| wichmann eb/syn (bulk) | none found | unstated (assume ~1.74) | no MRC on this filesystem; inferred from pipeline |

⚠️ **Metadata hygiene:** voxel-size units are stored inconsistently — some cooper files record `1.739`
(nm), others `17.36 / 17.39` (Ångström for the *same* 1.74 nm). The bulk wichmann `eb/syn` files store
**no `voxel_size` attribute at all**. Physically everything is ~1.74 nm, but the attribute can't be
trusted blindly.

## Data-addition timeline & what each step bought us

| date | change | effect on AP* |
|---|---|---|
| Feb–Mar 2026 | wichmann `inital_data` 115 files (eb/syn bulk) | baseline corpus |
| ~May 2026 | + cooper cristae (6) + wichmann Otof (13) | — |
| **2026-06-01** | **06-01 model**, 135-file split | **AP 0.837** (gold standard) |
| 2026-06-04 | + 5 high-res cooper tomograms (`36194 B4` series) | these are the **easiest** test cases (AP 0.93–0.98) |
| 2026-06-17 | **moved out 13 poorly-annotated files** | **+0.05 AP** (oldloss 0.780 → persample-aug 0.830) |

\* AP = threshold-free average precision within annotated mitochondria (`state==1`), on the pinned 15-file
test set.

**Key lesson — quality beat quantity.** Adding more wichmann/cooper volumes did *not* monotonically raise
AP (post-06-01 models sit at 0.77–0.83, driven by recipe, not data size). The single biggest *data* lever
was removing the 13 bad annotations (+0.05). Cleaning + per-sample masked Dice + Kornia augmentation +
~100k iters brings the cleaned extended set to **0.830**; 06-01's exact file list reproduces **0.837**.

### AP reference (15 pinned test files)

| model | AP | recipe |
|---|---|---|
| 06-01 (gold) | 0.837 | per-sample · aug · exact-120 split · ~100k |
| repro0601-exactsplit | 0.837 | reproduces 06-01 on current code |
| **bs24-AMP-allfiles** | **0.832** | per-sample · aug · allfiles · lr=1.7e-4 · **bs=24** · AMP bf16 · 100k |
| persample-aug | 0.830 | per-sample · aug · extended split · **bad files excluded** · 100k |
| newloss-aug | 0.822 | pooled loss · aug · extended · bad excluded · 75k |
| oldloss | 0.780 | per-sample · aug · extended · **bad files included** · 75k |
| repro0601-legacy | 0.767 | pooled · **no aug** · legacy |

## Pinned test set (keeps results comparable across runs)

`SYNAPSENETV1_TEST_SPLIT` in `training/cristae/train_cristae.py` — 11 wichmann + 4 cooper, held out of
every run (`test_split: synapsenetv1-testsplit`). Hardest: `Otof_D3` (~0.53), `M8_eb6` (~0.62); easiest:
the cooper tomograms (0.93–0.98). The weak files are annotation-quality limits, not recipe artifacts.

## What matters most (recommendations)

1. **Heavy class imbalance** — cristae ≈ **16.7%** of annotated-mito voxels, mito ≈ 4% of the volume. The
   masked per-sample Dice (ignoring `state==2` unannotated mito) is doing real work; keep it.
2. **Data quality is the active lever, not volume** — next gains likely come from re-annotating the 13
   `needs_corrections` files and the weak test cases, not from adding more tomograms.
3. **Genotype coverage is WT-skewed** (~88 WT/CTRL vs ~37 KO/DKO; 29 wichmann unlabeled) — record this if
   genotype-balanced evaluation ever matters.
4. **Fix voxel-size metadata** — consistent nm units + backfill the wichmann `eb/syn` files at 1.748 nm so
   inference-time resampling can rely on the attribute.

## Related docs
- `evaluation/cristae/DIAGNOSIS_why_0601_wins_WIP.md` — why 06-01 wins (AP diagnostics)
- `evaluation/cristae/LOSS_old_vs_new_cristae.md` — old vs new masked-Dice loss
- `configs/training/cristae/cristae_persample_aug.yaml` — current training recipe
