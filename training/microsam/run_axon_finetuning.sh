#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-6:00:00
#SBATCH --job-name=finetune-sam
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH --constraint=inet


source /user/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/micro-sam2

# ================ Define ALL parameters here ONCE ================
N_ITER=20000
PATCH_SHAPE="1 256 256"
BS=1
LR=1e-4
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/all_cutouts_s2_new"
DD2="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented_s2_new/"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
EARLY_STOPPING=10
MODEL_TYPE="vit_b"  # _em_organelles
EXPNAME="microsam-axons-${MODEL_TYPE}-bs${BS}-ps${PATCH_SIZE}"
# use this to continue training from given checkpoint
# CHECKPOINT="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-all-wocytonuc/best.pt"



python /user/freckmann15/u15205/synapse/training/microsam/finetune_microsam.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD} \
  --data_dir2 ${DD2} \
  --early_stopping ${EARLY_STOPPING} \
  --raw_key ${RAW_KEY} \
  --model_type ${MODEL_TYPE} \
  # --checkpoint_path ${CHECKPOINT}