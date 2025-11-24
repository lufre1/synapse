#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=12:00:00
#SBATCH --job-name=train-mito-net
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --constraint 80gb

PATCH_SHAPE="64 256 256"
BS=8
LR=1e-4
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="mitotomo-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-s4"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_mito_tomo.py \
  --experiment_name ${EXPNAME} \
  --data_dir /mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all \
  --n_iterations 150000 \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --feature_size 32 \
#  --with_batchrenorm
