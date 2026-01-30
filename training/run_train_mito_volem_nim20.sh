#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-06:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --job-name=mito-net
#SBATCH --constraint 80gb


source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=15000
PATCH_SHAPE="32 512 512"
BS=4
LR=1e-4
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/"
RAW_KEY="raw"
LABEL_KEY="labels/mitochondria"
SDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented/"
# TDD="/mnt/lustre-grete/usr/u12103/mitopaper/4007_split/final_h5/"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="volume-em-mito-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-withWT-refined"
EARLY_STOPPING=10
SAVE_DIR="/mnt/lustre-grete/usr/u15205/volume-em/models/"
# use this to continue training from given checkpoint
# CHECKPOINT="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-all-wocytonuc/best.pt"

# this can be used to evaluate WT 
# /mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/4009_z427_y1084_x1020.h5

# export CUDA_LAUNCH_BLOCKING=1
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/training/train_mito_generic.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD} \
  --early_stopping ${EARLY_STOPPING} \
  --raw_key ${RAW_KEY} \
  --label_key ${LABEL_KEY} \
  --second_data_dir ${SDD} \
  --use_synapse_training \
  # --third_data_dir ${TDD} \
  # --checkpoint ${CHECKPOINT}
