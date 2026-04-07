#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --job-name=axon-net-training
#SBATCH --constraint 80gb


source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=50000
PATCH_SHAPE="32 512 512"
BS=4
LR=5e-4
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/all_cutouts_s2_new"
SDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented_s2_new/"
TDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split_s2_new/4009"
RAW_KEY="raw"
LABEL_KEY="labels/axons"

PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="volume-em-axons-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-s2-4009"
EARLY_STOPPING=25
SAVE_DIR="/mnt/lustre-grete/usr/u15205/volume-em/models/"
# export CUDA_LAUNCH_BLOCKING=1
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/training/axons/train_axons_volem.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${SDD} \
  --early_stopping ${EARLY_STOPPING} \
  --raw_key ${RAW_KEY} \
  --label_key ${LABEL_KEY} \
  --save_dir ${SAVE_DIR} \
  --data_dir2 ${TDD} \
#   --use_synapse_training \
  # --checkpoint ${CHECKPOINT}
