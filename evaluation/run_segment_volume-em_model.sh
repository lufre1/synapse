#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 64G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
BLOCK_SHAPE="32 256 256"
# DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/raw_volume.h5"
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/4007_split/final_h5/"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/volume_em_segmentation_on_new_splits"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mitochondria/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x256x256-thinboundary-cutout1and2"
FILE_EXTENSION=".h5"
SEED_DISTANCE=3


python /user/freckmann15/u12103/synapse/evaluation/segment_mitochondria.py \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  -ts ${BLOCK_SHAPE} \
  --model_path ${MODEL_PATH} \
  -ak \
  --seed_distance ${SEED_DISTANCE}
#  --force_overwrite
