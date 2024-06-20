#!/bin/bash

#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --job-name=mito-net64
#SBATCH -c 64
#SBATCH --ntasks=1

source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/test.py \
  --experiment_name "mitotomo-net32-bs2-ps32448-lr1e-3-isotropicscaling-onmitotomo" \
  --patch_shape 32 448 448 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --feature_size 32 \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-bs2-ps32448-lr1e-3-isotropicscaling/best.pt \
  --file_path /scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/outer-membrane2/2_20230415_TOMO_HOI_WT_36859_J1_STEM750/36859_J1_STEM750_66K_SP_07_rec_2kb1dawbp_crop.h5



# srun /home/nimlufre/miniforge3/envs/synapse/bin/python /home/nimlufre/synapse/main.py \
#   --experiment_name mito-net-bs1-ps-48 \
#   --patch_shape 48 384 384 \ 448 also possible as well as 480
#   --batch_size 1 \
#  #SBATCH --gres=gpu:A100:1

