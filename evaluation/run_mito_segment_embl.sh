#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --nodes=1
#SBATCH --job-name=mito_segment
#SBATCH --constraint 80gb

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/micromamba/envs/synapse

python evaluation/segment_mitochondria.py \
    -b /scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_1/images/ome-zarr/raw.ome.zarr \
    -fe .zarr \
    -m '/user/freckmann15/u12103/synapse/models/mito-v3' \
    -k 0 \
    -e /scratch-grete/projects/nim00007/data/mitochondria/embl/out_embl