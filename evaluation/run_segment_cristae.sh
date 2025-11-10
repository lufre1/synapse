#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval-cristae-tomo
#SBATCH -c 4
#SBATCH --mem 32G


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
# cellmaps on volume em data
BLOCK_SHAPE="32 256 256"
DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/raw_volume.h5"
EXPORT_PATH="/scratch-grete/usr/nimlufre/synapse/mitotomo/test_segmentations_cristae/cristae-net32-lr1e-4-bs4-ps32x256x256-new_transform_corrected_tiling"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/cristae-net32-lr1e-4-bs4-ps32x256x256-new_transform"



python /user/freckmann15/u12103/synapse/evaluation/segment_cristae.py \
  --base_path ${DD} \
  --export_path ${EXPORT_PATH} \
  --model_path ${MODEL_PATH} \
  -ts ${BLOCK_SHAPE} \