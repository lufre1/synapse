#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=inference-microsam
#SBATCH -c 8
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/sam

# ================ Define ALL parameters here ONCE ================
BLOCK_SHAPE="1 256 256"
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr"
RAW_KEY="0"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
LD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/committed_objects_leonie_2025-08-07.tif"
LABEL_KEY="label_crop/all"
WITH_DISTANCES=True
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1_segmentation_microsam256"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/cellmap/checkpoints/microsam-cellmaps-vit_b_em_organelles-bs1-ps256-resized-wocytonucmem"
FILE_EXTENSION=".zarr"


python /user/freckmann15/u12103/synapse/evaluation/segment_with_sam.py \
  --block_shape ${BLOCK_SHAPE} \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  -ld ${LD} \
  --with_distances \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  # --model_path ${MODEL_PATH} \
  # --label_key ${LABEL_KEY} \
  # --force_override