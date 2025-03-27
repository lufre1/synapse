#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=run-cristae-net
#SBATCH --constraint 80gb

exp_name="cristae-net32-bs2-ps48512-cooper-wichmann-new-transform"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse/evaluation/segment_cristae.py \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/cristae-net32-bs2-ps48512-cooper-wichmann-new-transform \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined \
  --data_dir2 /scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2/ \
  --tile_size 48 512 512 \
  --halo 8 64 64 \
  --output_path /scratch-grete/usr/nimlufre/synapse/$exp_name/out

