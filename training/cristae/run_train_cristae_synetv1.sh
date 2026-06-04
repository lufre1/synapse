#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-16:00:00
#SBATCH --job-name=train-cristae-net
#SBATCH -c 8
#SBATCH --mem 64G

CONFIG="/user/freckmann15/u12103/synapse/training/cristae/cristae_net_synetv1.yaml"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/cristae/train_cristae.py \
  --config ${CONFIG}
