#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=train-mito-net
#SBATCH --constraint 80gb


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_mito_cellmap.py \
  --experiment_name "mitonet32-bs3-64256-lr1e-4-cellmap-normedraw" \
  --n_iterations 100000 \
  --patch_shape 64 256 256 \
  --batch_size 3 \
  --learning_rate 1e-4 \
  --feature_size 32 \
