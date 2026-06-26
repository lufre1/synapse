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
CUDA_VISIBLE_DEVICES=0 python /mnt/vast-nhr/home/freckmann15/u12103/synapse/training/cristae/train_cristae.py --experiment_name cristae-net32-lr1.4e-4-bs16-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26 --use_flashoptim --batch_size 16 --learning_rate 0.00014 --loss_variant persample --augmentations --split_strategy legacy --data_dir /scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2 --data_dir2 /mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae --data_dir3 /mnt/lustre-grete/usr/u12103/cristae_data/wichmann/ --data_dir4 /mnt/lustre-grete/usr/u12103/cristae_data/wichmann_needs_corrections/ --patch_shape 32 256 256 --n_iterations 100000 --feature_size 32 --ignore_state_value 2 --state_channel 1 --test_split synapsenetv1-testsplit --early_stopping 25 --seed 42 --num_workers 8 --persistent_workers --prefetch_factor 4

echo 'Done.'
