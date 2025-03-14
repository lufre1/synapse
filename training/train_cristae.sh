#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=mito-net

exp_name="cristae-net32-bs2-ps64512"

source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/train_cristae.py \
  --experiment_name $exp_name \
  --n_iterations 100000 \
  --batch_size 2 \
