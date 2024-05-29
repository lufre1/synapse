#!/bin/bash

#SBATCH --partition=grete:shared  # Requesting partition 'grete:shared'
#SBATCH --gres=gpu:A100:1          # Requesting 1 A100 GPU
#SBATCH --time=2-00:00:00         # Max wallclock time of 2 days
#SBATCH --account=nim00007         # Charge the job to account 'nim00007'
#SBATCH --job-name=mito-net     # Optional: Set a name for the job (replace with your desired name)

srun main.py \
#  --checkpoint_path /home/nimlufre/synapse \  # use only if weights are to be loaded
  --experiment_name mito-net-bs1-ps-48 \
  --patch_shape 48 384 384 \
  --batch_size 1 \


