#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval-mito-tomo
#SBATCH -c 4
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
# cellmaps on volume em data
BLOCK_SHAPE="32 512 512"
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/raw_volume.h5"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
EXPORT_PATH="/scratch-grete/usr/nimlufre/synapse/mitotomo/test_segmentations/mitotomo-net32-lr1e-4-bs1-ps32x512x512"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-lr1e-4-bs1-ps32x512x512-cooper-wichmann-new/"
FILE_EXTENSION=".h5"


python /user/freckmann15/u12103/synapse/evaluation/segment_mitochondria.py \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  --model_path ${MODEL_PATH} \
  -ts ${BLOCK_SHAPE} \
  -ak \
#  --force_overwrite