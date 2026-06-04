#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH --job-name=mito-net
#SBATCH --constraint 80gb


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=20000
PATCH_SHAPE="32 256 256"
BS=4
LR=1e-4
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/cutouts_h5"
RAW_KEY="0"
SDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/cutout_1_hdf5"
TDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/cutout_2_hdf5"
LABEL_KEY="labels/mitochondria"

# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="volume-em-mito-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-thinboundary-all_cutouts_luca"
EARLY_STOPPING=10
# use this to continue training from given checkpoint
# CHECKPOINT="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-all-wocytonuc/best.pt"


# export CUDA_LAUNCH_BLOCKING=1
python /user/freckmann15/u12103/synapse/training/train_mito_generic.py \
  --experiment_name "${EXPNAME}" \
  --n_iterations ${N_ITER} \
  --patch_shape ${PATCH_SHAPE} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD} \
  --early_stopping ${EARLY_STOPPING} \
  --raw_key ${RAW_KEY} \
  --second_data_dir ${SDD} \
  --label_key ${LABEL_KEY} \
  # --checkpoint ${CHECKPOINT}
