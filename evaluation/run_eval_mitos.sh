#!/bin/bash
#SBATCH -p standard96s:shared
#SBATCH --time=1-12:00:00
#SBATCH --job-name=eval-segment-param-grid
#SBATCH -c 16
#SBATCH --mem 128G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

H5_DIR="/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_s4_mito_refined_segmentations"

python /user/freckmann15/u12103/synapse/evaluation/evaluate_mitos.py \
    -l ${H5_DIR} \
    -k "labels/mitochondria" \
    -s ${H5_DIR} \
    -sk "seg" \
    -o ${H5_DIR} \
    -le ".h5" \
    -se ".h5"
