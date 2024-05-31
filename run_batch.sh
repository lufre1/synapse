#!/bin/bash

#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --job-name=mito-net-bs1-ps-64-test
#SBATCH -c 16
#SBATCH --ntasks=1

source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/main.py \
  --experiment_name "mito-net-bs1-ps-64-test" \
  --patch_shape 64 512 512 \
  --batch_size 1
#  --checkpoint_path /home/nimlufre/synapse \  # use only if weights are to be loaded



# srun /home/nimlufre/miniforge3/envs/synapse/bin/python /home/nimlufre/synapse/main.py \
#   --experiment_name mito-net-bs1-ps-48 \
#   --patch_shape 48 384 384 \ 448 also possible
#   --batch_size 1 \
# 
