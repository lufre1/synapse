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
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/inference/cristae/segment_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae_membrane_weighted_wneg2.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae_membrane_weighted_wneg2.yaml && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae_ap.py -c /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/evaluation/cristae/eval_cristae_membrane_weighted_wneg2.yaml

echo 'Done.'
