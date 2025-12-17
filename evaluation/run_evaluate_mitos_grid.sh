#!/bin/bash
#SBATCH -p standard96s:shared
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-segment-param-grid
#SBATCH -c 16
#SBATCH --mem 128G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/evaluation/evaluate_mitos_grid.py \
  -l "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/volem_seg_parameter/" \
  -le ".h5" \
  -k "labels/mitochondria" \
  -s "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/volem_seg_parameter/" \
  -se ".h5" \
  -o "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/volem_seg_parameter/"