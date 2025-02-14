#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem 32G
#SBATCH --job-name=train-cristae-net
#SBATCH --constraint 80gb

exp_name="cristae-net32-bs2-ps48512-cooper-wichmann-new-transform"

source /user/freckmann15/u12103/.bashrc
micromamba activate synapse

python /user/freckmann15/u12103/synapse/train_cristae_new.py \
  --experiment_name $exp_name \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined \
  --data_dir2 /scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2/ \
  --n_iterations 100000 \
  --batch_size 2 \
  --patch_shape 48 512 512 \
  --learning_rate 1e-4 \
  --feature_size 32 \
