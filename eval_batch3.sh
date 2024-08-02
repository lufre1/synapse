#!/bin/bash

#SBATCH --partition=grete-h100:shared
#SBATCH -G H100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --job-name=mito-net32
#SBATCH -c 64
#SBATCH --ntasks=1

exp_name="mitotomo-net32-bs1-ps64512-lr1e-4-scaling122-withrois-fididata"

source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/test.py \
  --experiment_name $exp_name \
  --down_scale_factor 2 \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/$exp_name \
  --file_path /scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/outer-membrane2/2_20230415_TOMO_HOI_WT_36859_J1_STEM750/36859_J1_STEM750_66K_SP_07_rec_2kb1dawbp_crop.h5


