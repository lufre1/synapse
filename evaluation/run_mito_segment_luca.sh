#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --job-name=eval-mito-net
#SBATCH -c 8
#SBATCH --mem 64G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
TILE_SHAPE="32 256 256"
# DD="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_2/images/ome-zarr/raw.ome.zarr"
DD="/scratch-grete/projects/nim00007/data/mitochondria/embl/4007/images/ome-zarr/raw.ome.zarr"
RAW_KEY="0"
LABEL_DATA="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_2/images/ome-zarr/mitos.ome.zarr"
LABEL_KEY="0"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/test_segmentations/all_256_thinboundary_luca_small_rand"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mitochondria/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x256x256-thinboundary-cutout1and2"
SEED_DISTANCE=2
BOUNDARY_THRESHOLD=0.25
ALL_KEYS=True

python /user/freckmann15/u12103/synapse/evaluation/segment_mitochondria.py \
  -b ${DD} \
  -e ${EXPORT_PATH} \
  -m ${MODEL_PATH} \
  -k ${RAW_KEY} \
  -ts ${TILE_SHAPE} \
  -sd ${SEED_DISTANCE} \
  -bt ${BOUNDARY_THRESHOLD} \
  -fo \
  -ak 
  # -lp ${LABEL_DATA} \
  # -lk ${LABEL_KEY} \
