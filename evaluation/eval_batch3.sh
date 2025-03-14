#!/bin/bash

#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --job-name=mito-net32
#SBATCH -c 8
#SBATCH --ntasks=1

exp_name="mitotomo-net32-bs2-ps64512-lr1e-4-all-mitos"

source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/test.py \
  --experiment_name $exp_name \
  --down_scale_factor 2 \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/$exp_name \
  --patch_shape 64 512 512 \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/training_blocks_v1/ 


