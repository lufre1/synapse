#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 64G
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
BLOCK_SHAPE="32 512 512"
RAW_KEY="raw"
LABEL_KEY="labels/mitochondria"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/volem_seg_parameter/"
# FORCE_OVERRIDE=True
MODEL_PATH="/mnt/lustre-grete/usr/u12103/mitochondria/volem/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x512x512-withWT"
# MODEL_PATH=" /scratch-grete/usr/nimlufre/synapse/mitochondria/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x256x256-thinboundary-cutout1and2/"
FILE_EXTENSION=".h5"
SEED_DISTANCE=3
DD="/mnt/lustre-grete/usr/u12103/mitochondria/moebius/4007/train_split/cutout_2/raw.ome_s1.h5"


python /user/freckmann15/u12103/synapse/scripts/find_segmentation_parameter.py \
  --key ${RAW_KEY} \
  --label_key ${LABEL_KEY} \
  --export_path ${EXPORT_PATH} \
  --file_extension ${FILE_EXTENSION} \
  -ts ${BLOCK_SHAPE} \
  --model_path ${MODEL_PATH} \
  --base_path ${DD}
