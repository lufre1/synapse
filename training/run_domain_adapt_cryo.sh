#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --job-name=fidi_mito_segment
#SBATCH --constraint 80gb

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse/training/domain_adapt_mito_cryoet.py \
    --experiment_name "domain-adapt-cryoet-ps32512-bs1" \
    --data_dir /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/ \
    --checkpoint_path "/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-bs2-ps32512-lr1e-4-downscaled" \
    --confidence_threshold 0.5 \
    --n_iterations 30000 \
    --batch_size 1 \
    --patch_shape 32 512 512 
