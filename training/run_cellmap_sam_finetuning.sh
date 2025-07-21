#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cellmap-sam
#SBATCH -c 8


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/sam

# ================ Define ALL parameters here ONCE ================
N_ITER=13000
PATCH_SHAPE="1 256 256"
BS=1
LR=1e-4
DD="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
EXPNAME="microsam-cellmaps-bs${BS}-ps${PATCH_SIZE}-resized-all"



python /user/freckmann15/u12103/synapse/training/train_organelle_groups_sam.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD}