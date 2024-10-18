#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH --job-name=domain-adapt-mito
#SBATCH --constraint 80gb


source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/train_mito_domain_adaptation.py \
  --experiment_name "mito-domain-adapt-s2-sampler" \
  --data_dir /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/ \
  --checkpoint_path "/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-bs4-ps32256-lr1e-4-downscaled/" #\
  # --n_iterations 100000 \
  # --patch_shape 32 256 256 \

# mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held_s2/