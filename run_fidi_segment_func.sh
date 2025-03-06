#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem 32G
#SBATCH --job-name=fidi_mito_segment
#SBATCH --constraint 80gb

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse-net/scripts/cooper/run_mitochondria_segmentation.py\
  -i /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250228_for_Mito_Seg/ \
  -o /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250228_for_Mito_Seg/out\
  --tile_shape 48 512 512 \
  --halo 4 128 128
