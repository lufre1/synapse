#!/bin/bash
#SBATCH --time=0-24:00:00
#SBATCH --job-name=grid-search-mitos-4007
#SBATCH -c 8
#SBATCH --mem 512G
#SBATCH --partition=large96s
##SBATCH --partition=standard96s:shared

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

SYNAPSE="/mnt/vast-nhr/home/freckmann15/u15205/synapse"
CONFIG="$SYNAPSE/inference/mitochondria/grid_search_4007.yaml"

python $SYNAPSE/inference/mitochondria/grid_search_mitos_ooc.py \
  --config $CONFIG \
  --seed_distance     1 2 3 \
  --bg_penalty        1.0 1.5 2.5 \
  --foreground_threshold 0.5 0.6 0.7 \
  --boundary_threshold   0.05 0.08 0.12
