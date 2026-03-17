#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-02:00:00
#SBATCH --qos 2h
#SBATCH --job-name=inference-volume-em
#SBATCH -c 8
#SBATCH --mem 64G
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

CONFIG_FILE_PATH="/mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/axons/segment_axons_4009.yaml"

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/axons/segment_axons_ooc.py \
  --config $CONFIG_FILE_PATH

