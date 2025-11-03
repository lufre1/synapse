#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=train-mito-net

PATCH_SHAPE="32 256 256"
BS=4
LR=1e-4
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="mitotomo-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-cooper-wichmann-new"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_mito_tomo.py \
  --experiment_name ${EXPNAME} \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos \
  --data_dir2 /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/ \
  --data_dir3 /mnt/lustre-grete/usr/u12103/mitochondria/cooper/fidi_2025/exported_to_hdf5_s2 \
  --n_iterations 150000 \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --feature_size 32 \
