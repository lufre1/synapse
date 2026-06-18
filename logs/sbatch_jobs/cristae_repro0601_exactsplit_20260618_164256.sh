#!/bin/bash
#SBATCH --job-name=train_cristae
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1-12:00:00
#SBATCH --nodes 1
#SBATCH -p grete:shared
#SBATCH -G A100:1
source ~/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse
CUDA_VISIBLE_DEVICES=0 python /mnt/vast-nhr/home/freckmann15/u12103/synapse/training/cristae/train_cristae.py --experiment_name cristae-net32-lr1e-4-bs8-ps32x256x256-repro0601-exactsplit-2026-06-18 --loss_variant persample --augmentations --split_file /mnt/vast-nhr/home/freckmann15/u12103/synapse/configs/training/cristae/splits/cristae_0601_exact_split.json --patch_shape 32 256 256 --n_iterations 75000 --batch_size 8 --learning_rate 0.0001 --feature_size 32 --ignore_state_value 2 --state_channel 1 --early_stopping 30 --seed 42 --num_workers 8 --persistent_workers --prefetch_factor 4

echo 'Done.'
