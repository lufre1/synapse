#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --job-name=eval-mito-net
#SBATCH -c 8
#SBATCH --mem 64G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
TILE_SHAPE="48 512 512"
# DD="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_2/images/ome-zarr/raw.ome.zarr"
DD="/scratch-grete/projects/nim00007/data/mitochondria/embl/4007/images/ome-zarr/raw.ome.zarr"
RAW_KEY="raw"
# LABEL_DATA="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_2/images/ome-zarr/mitos.ome.zarr"
LABEL_KEY="labels/mitochondria"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cooper-wichmann-model"
FORCE_OVERRIDE=True
MODEL_PATH="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitonet32-bs2-ps48512-lr1e-4-combined-thick"
SEED_DISTANCE=6
BOUNDARY_THRESHOLD=0.2
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