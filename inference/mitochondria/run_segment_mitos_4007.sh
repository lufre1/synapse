#!/bin/bash
#SBATCH --time=0-2:30:00
#SBATCH --job-name=inference-volume-em-4007
#SBATCH -c 8
#SBATCH --mem 256G
#SBATCH --partition=standard96s:shared

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

CONFIG_FILE_PATH="/mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitos_4007.yaml"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitochondria_ooc.py \
  --config $CONFIG_FILE_PATH
