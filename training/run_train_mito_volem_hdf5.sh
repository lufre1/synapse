#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --job-name=mito-net
#SBATCH --constraint 80gb


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=15000
PATCH_SHAPE="32 512 512"
BS=4
LR=1e-4
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/cutout_1/"
RAW_KEY="raw"
LABEL_KEY="labels/mitochondria"
SDD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/cutout_2/"
TDD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/final_h5/"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="volume-em-mito-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-all"
EARLY_STOPPING=20
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
  --label_key ${LABEL_KEY} \
  --second_data_dir ${SDD} \
  --third_data_dir ${TDD} \
  --use_synapse_training
  # --checkpoint ${CHECKPOINT}
