#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem 128G
#SBATCH --job-name=train-mito-net32
#SBATCH --constraint 80gb


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse/training/train_mito_generic.py \
  --experiment_name "mitonet32-bs2-ps48512-lr1e-4-combined-bcedice" \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos \
  --data_dir2 /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/ \
  --n_iterations 100000 \
  --patch_shape 48 512 512 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 32 \
  --without_rois 1 \
  --early_stopping 15
