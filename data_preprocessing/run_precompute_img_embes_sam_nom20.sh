#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-01:00:00
#SBATCH --job-name=precompute-embeddings-sam
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -C inet
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/micro-sam2

# ================ Define ALL parameters here ONCE ================

# DD="/mnt/lustre-grete/usr/u15205/mobie/project_4009/4009/images/ome-zarr/raw.ome.zarr"
DD="/mnt/lustre-grete/usr/u15205/mobie/project_4010/4010/images/ome-zarr/raw.ome.zarr"
EXPORT_PATH="/mnt/lustre-grete/usr/u15205/volume-em/4010/axon_embeddings/"
# MODEL_TYPE="vit_b_em_organelles"
MODEL_TYPE="vit_b"
CHECKPOINT_PATH="/mnt/lustre-grete/usr/u15205/volume-em/microsam/checkpoints/vit_b-axons-vit_b-bs1-ps256/best.pt"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/data_preprocessing/precompute_embeddings_with_sam.py \
  -b ${DD} \
  -o ${EXPORT_PATH} \
  -mt ${MODEL_TYPE} \
  --key s2 \
  -cp ${CHECKPOINT_PATH} \
  # --tile_shape 1 512 512 \
  # --halo 1 64 64 \