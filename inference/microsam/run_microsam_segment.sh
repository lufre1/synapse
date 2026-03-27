#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-1:00:00
#SBATCH --job-name=precompute-embeddings-sam
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -C inet
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/micro-sam2

# ================ Define ALL parameters here ONCE ================

DD="/mnt/lustre-grete/usr/u15205/mobie/project_4009/4009/images/ome-zarr/raw.ome.zarr"
SP="/mnt/lustre-grete/usr/u15205/mobie/project_4009/4009/images/ome-zarr/axons.ome.zarr"
EXPORT_PATH="/mnt/lustre-grete/usr/u15205/mobie/4009_microsam_segmentation"
# MODEL_TYPE="vit_b_em_organelles"
MODEL_TYPE="vit_b"
CHECKPOINT_PATH="/mnt/lustre-grete/usr/u15205/volume-em/microsam/checkpoints/vit_b-axons-vit_b-bs1-ps256/best.pt"
EMBEDDING_PATH="/mnt/lustre-grete/usr/u15205/mobie/4009_embeddings_tiled"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/microsam/microsam_segment.py \
  -i ${DD} \
  -s ${SP} \
  -e ${EXPORT_PATH} \
  -m ${MODEL_TYPE} \
  -ep ${EMBEDDING_PATH} \
  -cp ${CHECKPOINT_PATH} \
  --key s2 \
  -segk s2 \
