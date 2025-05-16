#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cellmap


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_mito_cellmap.py \
  --experiment_name "mitonet32-bs4-128-lr1e-4-cellmap-medium-organelles" \
  --n_iterations 100000 \
  --patch_shape 128 128 128 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --feature_size 32 \
