#!/bin/bash

#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --job-name=mito-net64
#SBATCH -c 16
#SBATCH --ntasks=1

source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/test.py \
  --experiment_name "mito-net32-bs2-ps32448-lr1e-3-isotropicscaling" \
  --patch_shape 32 256 256 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --feature_size 32 \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mito-net32-bs2-ps32448-lr1e-3-isotropicscaling-labelthreshold/best.pt



# srun /home/nimlufre/miniforge3/envs/synapse/bin/python /home/nimlufre/synapse/main.py \
#   --experiment_name mito-net-bs1-ps-48 \
#   --patch_shape 48 384 384 \ 448 also possible as well as 480
#   --batch_size 1 \
# 

