#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 06:00:00
#SBATCH --nodes 1
#SBATCH -p grete:interactive
#SBATCH -G 1g.20gb:1
source ~/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/repair_flashoptim_checkpoint.py --src /mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26 --out /mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26-repaired && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/inference/cristae/segment_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae_ap.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26.yaml

echo 'Done.'
