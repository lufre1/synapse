#!/bin/bash
## SBATCH --partition=grete:interactive
## SBATCH -G 1g.20gb:1
#SBATCH --time=0-5:00:00
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH --partition=grete:shared
#SBATCH -G A100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

CONFIG_FILE_PATH="/mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitos_4016.yaml"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitochondria_ooc.py \
  --config $CONFIG_FILE_PATH
