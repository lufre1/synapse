#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --job-name=mito-aniso2lvl
#SBATCH --constraint 80gb


source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
N_ITER=50000
PATCH_SHAPE="64 256 256"
BS=8
LR=1e-4
# Scale factors: 2 anisotropic [1,2,2] levels + 2 isotropic [2,2,2] levels
# Matches 5:1 z:xy anisotropy (z=25nm, xy=5nm)
SF="1 2 2 1 2 2 2 2 2 2 2 2"
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/"
RAW_KEY="raw"
LABEL_KEY="labels/mitochondria"
SDD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented/"
PATCH_SIZE=$(echo $PATCH_SHAPE | awk '{print $2}')
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="volume-em-mito-aniso2lvl-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-final"
EARLY_STOPPING=20
SAVE_DIR="/mnt/lustre-grete/usr/u15205/volume-em/models/"

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
  --scale_factors ${SF} \
  --mixed_precision \
  --use_synapse_training \
  # --checkpoint ${CHECKPOINT}
