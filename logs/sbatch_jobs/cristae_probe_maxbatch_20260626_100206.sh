#!/bin/bash
#SBATCH --job-name=probe_max_batch
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 1-12:00:00
#SBATCH --nodes 1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --constraint 80gb
source ~/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse
CUDA_VISIBLE_DEVICES=0 python /mnt/vast-nhr/home/freckmann15/u12103/synapse/training/cristae/probe_max_batch.py --batch_sizes 8,12,16,20,24,28,32,40,48 --patch_shape 32 256 256 --feature_size 32 --regimes fp32,amp --iters 3

echo 'Done.'
