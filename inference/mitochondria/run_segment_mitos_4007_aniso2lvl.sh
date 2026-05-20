#!/bin/bash
#SBATCH --time=0-2:30:00
#SBATCH --job-name=inference-volume-em-4007-aniso2lvl
#SBATCH -c 8
#SBATCH --mem 256G
#SBATCH --partition=grete-h100:shared
#SBATCH -G H100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

CONFIG_FILE_PATH="/mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitos_4007_aniso2lvl.yaml"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitochondria_ooc.py \
  --config $CONFIG_FILE_PATH
