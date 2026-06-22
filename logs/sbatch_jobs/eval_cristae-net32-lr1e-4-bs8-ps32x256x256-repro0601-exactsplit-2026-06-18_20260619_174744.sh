#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1-12:00:00
#SBATCH --nodes 1
#SBATCH -p grete:shared
#SBATCH -G A100:1
source ~/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/inference/cristae/segment_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1e-4-bs8-ps32x256x256-repro0601-exactsplit-2026-06-18.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1e-4-bs8-ps32x256x256-repro0601-exactsplit-2026-06-18.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae_ap.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae-net32-lr1e-4-bs8-ps32x256x256-repro0601-exactsplit-2026-06-18.yaml

echo 'Done.'
