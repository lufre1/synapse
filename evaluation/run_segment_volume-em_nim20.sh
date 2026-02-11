#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 64G
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
BLOCK_SHAPE="32 512 512"
# DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/raw_volume.h5"
DD="/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split/"
# DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/final_h5/4007_block_z001024_001152_y001936_003872_x003312_004968.h5"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
EXPORT_PATH="/mnt/lustre-grete/usr/u15205/volume-em/test_split_segmentations_final"
# FORCE_OVERRIDE=True
MODEL_PATH="/mnt/lustre-grete/usr/u15205/volume-em/models/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x512x512-final"
# MODEL_PATH=" /scratch-grete/usr/nimlufre/synapse/mitochondria/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x256x256-thinboundary-cutout1and2/"
FILE_EXTENSION=".h5"
SEED_DISTANCE=2


python /mnt/vast-nhr/home/freckmann15/u15205/synapse/evaluation/segment_mitochondria.py \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  -ts ${BLOCK_SHAPE} \
  --model_path ${MODEL_PATH} \
  -ak \
  --seed_distance ${SEED_DISTANCE} \
  -ft 0.6 \
  -bt 0.1 \
  -at 200 \
  -uc \
  --post_iter3d 0 \
  -pv \
  -ms 1000 \
  -fo
  # -cc \
  # -de 2 \
#  --force_overwrite
