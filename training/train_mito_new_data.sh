#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem 128G
#SBATCH --job-name=train-mito-net32
#SBATCH --constraint 80gb
#SBATCH -o /user/freckmann15/u12103/outfiles/synapse/outfile-%J


source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/main.py \
  --experiment_name "mitotomo-net32-bs2-ps32512-lr1e-4-downscaled" \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2 \
  --n_iterations 100000 \
  --patch_shape 32 512 512 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 32 \
  --without_rois 1