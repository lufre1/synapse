# synapse-net cristae3 vs cristae4 — test-split comparison (2026-06-26)

## TL;DR
On the pinned cristae test split, **cristae3 and cristae4 are tied on Dice (both 0.760)**. They differ
only in the precision/recall operating point: **cristae3 is more precise** (0.80 P / 0.73 R — slightly
under-segments), **cristae4 is more balanced with higher recall** (0.77 P / 0.76 R). Neither dominates.

## Result — masked to annotated mito (state == 1), 14 files

| model | n | Dice | precision | recall | HD95 |
|---|---|---|---|---|---|
| **cristae3** (1.44 nm) | 14 | **0.7596** | 0.8031 | 0.7255 | — |
| **cristae4** (1.74 nm) | 14 | **0.7595** | 0.7662 | 0.7582 | — |

Dice/precision/recall: higher = better. (Per-file CSVs:
`…/test_segmentations/synapsenet_cli/eval_cristae{3,4}/cristae_eval_results.csv`.)

### Per-file Dice
| file | cristae3 | cristae4 |
|---|---|---|
| 36194_B4_…_SC_01 (cooper) | 0.934 | 0.915 |
| 36194_B4_…_SC_22 (cooper) | 0.936 | 0.909 |
| 36859_J1_…_PS_03 (cooper) | 0.917 | 0.866 |
| KO8_eb4 | 0.720 | 0.735 |
| M1_eb6 | 0.717 | 0.783 |
| M2_eb5 | 0.741 | 0.720 |
| M5_eb1 | 0.762 | 0.780 |
| M8_eb6 | 0.622 | 0.588 |
| Otof_AVCN03_429A | 0.560 | 0.543 |
| WT20_eb5 | 0.760 | 0.788 |
| WT21_eb5 | 0.779 | 0.783 |
| WT21_syn4 | 0.713 | 0.734 |
| WT22_eb5 | 0.713 | 0.740 |
| WT40_eb10 | 0.760 | 0.750 |

Pattern: both segment the **cooper** tomograms well (Dice 0.87–0.94); the **wichmann** files are harder
(0.54–0.79). cristae3 wins on the cooper files (its higher precision pays off there); cristae4 edges
ahead on several wichmann files via higher recall. Worst case for both: `Otof_AVCN03_429A` (~0.55).

## Method
- Ran the **synapse-net `run_segmentation` CLI** (`synapse_net` 0.5.0) on each test file's combined h5
  (`raw_mitos_combined` [2,Z,Y,X] → ch0 raw, ch1 mito). Job `14509075`; one process per file.
- **Per-model scale** (rescale data to each model's training resolution): cristae3 trained @1.44 nm →
  `--scale 1.208` (data is ~1.74 nm); cristae4 trained @1.74 nm → `--scale 1.0` (native, no rescale).
- Scored with `evaluate_cristae.py` → masked to annotated mito (state == 1), exactly like the AP eval.
- Config: `configs/evaluation/cristae/eval_synapsenet_cristae3_vs_cristae4.yaml`.

## Caveats
- **No AP here.** The CLI emits a post-processed *segmentation*, not foreground probabilities, so only
  thresholded metrics (Dice/precision/recall) are available — not the threshold-free AP used elsewhere
  (06-01 = 0.837). Numbers are therefore **not** comparable to the AP table; compare them only to each
  other (and to other *segmentation*-Dice numbers under the same post-processing).
- **14, not 15, files.** `Otof_AVCN07_455L_…_B3_2` (the needs_corrections file) was excluded: its
  `raw_mitos_combined` is stored as **float32** (others int16), which crashes synapse-net's
  `_erode_instances → regionprops` ("Non-integer label_image") identically for both models — a
  data-format issue, not model quality.
- **HD95 not computed** (ran without `--compute_hd95` for speed). Re-run the eval with `-hd` if a
  boundary metric is wanted.
