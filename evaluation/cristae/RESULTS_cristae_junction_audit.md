# Cristae → mito-membrane contact points: data audit

**Question:** do we have enough cristae labels that show cristae junctions (contact points) to the mitochondrial membrane? Junctions are *not* annotated — they are computed with the synapse-net cristae-widget recipe: the membrane is approximated as the mito mask minus its erosion (an N-nm shell) from `raw_mitos_combined[1]==1` (state-1, cristae-annotated mitochondria), and a contact point is a connected component of `cristae & membrane`.

Files audited: **147** (147-file training discovery set). Membrane thickness swept at **8, 12 nm**. Files skipped (no cristae or no state-1 mito): 0.

## Corpus totals

| thickness_nm | files_analyzable | files_with_junction | total_contact_points | total_cristae_instances | cristae_touching | pct_cristae_touching | total_mito_instances | mito_with_junction | pct_mito_with_junction | total_contact_voxels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.0 | 147 | 125 | 2119 | 2341 | 994 | 42.5 | 315 | 228 | 72.4 | 5351987 |
| 12.0 | 147 | 143 | 2802 | 2341 | 1163 | 49.7 | 315 | 266 | 84.4 | 6791604 |

## Distributions

- **8 nm** — contact points per file: median=10 IQR=[4,19] max=123; per mito-instance: median=1 IQR=[1,2] max=6; crista voxels in contact: median 2.0%
- **12 nm** — contact points per file: median=14 IQR=[6,24] max=123; per mito-instance: median=2 IQR=[1,2] max=6; crista voxels in contact: median 2.2%

## By data source

| thickness_nm | source | files | files_with_junction | contact_points | cristae_inst | cristae_touching | mito_inst | mito_with_junction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.0 | cooper | 22 | 18 | 451 | 762 | 231 | 74 | 40 |
| 8.0 | wichmann | 125 | 107 | 1668 | 1579 | 763 | 241 | 188 |
| 12.0 | cooper | 22 | 22 | 648 | 762 | 294 | 74 | 53 |
| 12.0 | wichmann | 125 | 121 | 2154 | 1579 | 869 | 241 | 213 |

## By genotype (heuristic from filename)

| thickness_nm | genotype | files | files_with_junction | contact_points | cristae_inst | cristae_touching | mito_inst | mito_with_junction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.0 | DKO | 1 | 1 | 5 | 8 | 2 | 1 | 1 |
| 8.0 | KO | 22 | 9 | 185 | 465 | 71 | 56 | 16 |
| 8.0 | WT | 95 | 87 | 1475 | 1434 | 669 | 197 | 152 |
| 8.0 | unspecified | 29 | 28 | 454 | 434 | 252 | 61 | 59 |
| 12.0 | DKO | 1 | 1 | 5 | 8 | 2 | 1 | 1 |
| 12.0 | KO | 22 | 21 | 334 | 465 | 120 | 56 | 37 |
| 12.0 | WT | 95 | 92 | 1908 | 1434 | 778 | 197 | 168 |
| 12.0 | unspecified | 29 | 29 | 555 | 434 | 263 | 61 | 60 |

## Thickness sensitivity

- At 8 nm, 22 analyzable files have **zero** contact points; of those, 18 gain ≥1 at 12 nm. Large 8→12 nm swings mean those cristae sit just inside the eroded shell — the contact count is sensitive to the assumed membrane thickness for those files.

## Caveats

- Membrane is an **approximation** (erosion shell), not a real IMM/OMM segmentation; contact counts scale with the assumed thickness.
- A single crista can touch the membrane in several disconnected patches, so `contact_points` (connected components) can exceed the number of cristae instances.
- Voxel size is read from the file attr when present, else assumed **1.74 nm isotropic** (all wichmann files); the nm→voxel conversion of the shell thickness depends on it.
- Genotype is parsed heuristically from filenames (WT / KO / DKO / unspecified).
