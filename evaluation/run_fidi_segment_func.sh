#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --job-name=fidi_mito_segment
#SBATCH --constraint 80gb

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse-net/scripts/cooper/run_mitochondria_segmentation.py\
  -i /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/done \
  -o /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/fidi_script_seg__model2_out \
  # --tile_shape 48 512 512 \
  # --halo 8 128 128
