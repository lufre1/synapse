#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=train-cristae-net

PATCH_SHAPE="32 256 256"
BS=4
LR=1e-4
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="cristae-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-new_transform"
DD1="/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2"
DD2="/mnt/lustre-grete/usr/u12103/mitochondria/cooper/fidi_2025/raw_mitos_combined_s2"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_cristae.py \
  --experiment_name ${EXPNAME} \
  --patch_shape ${PATCH_SHAPE} \
  --n_iterations 100000 \
  --batch_size 4 \
  --learning_rate ${LR} \
  --data_dir ${DD1} \
  --data_dir2 ${DD2}
