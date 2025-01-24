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


source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/train_mito_wichmann.py \
  --experiment_name "mitotomo-net32-bs2-ps48512-lr1e-4-wichmann-more-fully-annotated" \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/wichmann/more_fully_annotated_mitos \
  --n_iterations 100000 \
  --patch_shape 48 512 512 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 32 \
  --without_rois 1 \
  --early_stopping 15