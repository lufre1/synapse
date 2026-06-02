#!/bin/bash
#SBATCH --partition=grete-h100:shared
#SBATCH -G H100:1
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --job-name=pred-mitos-4007

source /mnt/vast-nhr/home/freckmann15/u15205/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u15205/micromamba/envs/synapse

python /mnt/vast-nhr/home/freckmann15/u15205/synapse/inference/mitochondria/predict_mitos_4007.py
