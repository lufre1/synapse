#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --job-name=watershed-4007-aniso2lvl-cpu
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --partition=large96s

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

CONFIG_FILE_PATH="/mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitos_4007_aniso2lvl_cpu.yaml"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/segment_mitochondria_ooc.py \
  --config $CONFIG_FILE_PATH
