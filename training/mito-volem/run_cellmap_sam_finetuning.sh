#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=cellmap-sam
#SBATCH -c 8
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/sam

# ================ Define ALL parameters here ONCE ================
N_ITER=20000
PATCH_SHAPE="1 256 256"
BS=1
LR=1e-4
DD="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
EXPNAME="microsam-cellmaps-vit_b_em_organelles-bs${BS}-ps${PATCH_SIZE}-resized-wocytonucmem-several-maps"
EARLY_STOPPING=10
MODEL_TYPE="vit_b_em_organelles"
# use this to continue training from given checkpoint
CHECKPOINT="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-all-wocytonuc/best.pt"



python /user/freckmann15/u12103/synapse/training/train_organelle_groups_sam.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD} \
  --early_stopping ${EARLY_STOPPING} \
  --raw_key ${RAW_KEY} \
  --model_type ${MODEL_TYPE} \
  # --checkpoint_path ${CHECKPOINT}