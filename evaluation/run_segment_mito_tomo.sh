#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval-mito-tomo
#SBATCH -c 6
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
# cellmaps on volume em data
BLOCK_SHAPE="32 256 256"
DD="/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_s4/"
# DD="/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all/wichmann/"
RAW_KEY="raw"
# DD="/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops"
# RAW_KEY="raw_crop"
# EXPORT_PATH="/scratch-grete/usr/nimlufre/synapse/mitotomo/wichmann_s4_segmentations"
EXPORT_PATH="/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_s4_mito_refined_segmentations_new_seg_func"
FORCE_OVERRIDE=True
# MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4/"
MODEL_PATH="/mnt/lustre-grete/usr/u12103/mitochondria/tomo/checkpoints/mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4-refined"
FILE_EXTENSION=".h5"
SEED_DISTANCE=4


python /user/freckmann15/u12103/synapse/evaluation/segment_mitochondria.py \
  --base_path ${DD} \
  --key ${RAW_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  --model_path ${MODEL_PATH} \
  -ts ${BLOCK_SHAPE} \
  -ak \
  -sd ${SEED_DISTANCE} \
  -ft 0.8 \
  -bt 0.1 \
  -uc
  # -am  # add missing mitos
#  --force_overwrite