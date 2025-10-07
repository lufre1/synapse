#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 128G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
# cellmaps on volume em data
BLOCK_SHAPE="128 128 128"
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/raw_volume.h5"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cellmap-mito_volume_em_segmentation_on_mitonet"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/cellmap/checkpoints/net32-bs8-ps128-lr1e-4-cellmap-mito"
FILE_EXTENSION=".h5"


python /user/freckmann15/u12103/synapse/evaluation/segment_mitochondria.py \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  --model_path ${MODEL_PATH} \
  -ts ${BLOCK_SHAPE} \
  -ak \
  --force_overwrite