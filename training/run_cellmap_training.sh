#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cellmap


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=100000
PATCH_SHAPE="256 256 256"
BS=1
LR=1e-4
FS=32
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $1}')
EXPNAME="net${FS}-bs${BS}-ps${PATCH_SIZE}-lr${LR}-cellmap-erwes"


python /user/freckmann15/u12103/synapse/training/train_organelle_group_cellmap.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --feature_size ${FS}

# python /user/freckmann15/u12103/synapse/training/train_organelle_group_cellmap.py \
#   --experiment_name "net32-bs1-256-lr1e-4-cellmap-vesiclesandendosomes" \
#   --n_iterations 100000 \
#   --patch_shape 256 256 256 \
#   --batch_size 1 \
#   --learning_rate 1e-4 \
#   --feature_size 32 \
