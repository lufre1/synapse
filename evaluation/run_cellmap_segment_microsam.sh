#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=cellmap-sam
#SBATCH -c 8
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/sam

# ================ Define ALL parameters here ONCE ================
BLOCK_SHAPE="1 256 256"
DD="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
LABEL_KEY="label_crop/all"
WITH_DISTANCES=True
EXPORT_PATH="/scratch-grete/usr/nimlufre/cellmap/test_segmentations_microsam-cellmaps-vit_b_em_organelles-bs1-ps256-resized-wocytonucmem/"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-resized-wocytonucmem"


python /user/freckmann15/u12103/synapse/evaluation/segment_with_microsam.py \
  --block_shape ${BLOCK_SHAPE} \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --label_key ${LABEL_KEY} \
  --with_distances \
  --export_path ${EXPORT_PATH} \
  --model_path ${MODEL_PATH} \
  # --force_override