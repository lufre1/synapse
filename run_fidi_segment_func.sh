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
micromamba activate synapse

python /user/freckmann15/u12103/synapse-net/scripts/cooper/run_mitochondria_segmentation.py\
  -i /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/20250212_test_I/ \
  -o /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/20250212_test_O/ \
  --tile_shape 48 512 512 \
  --halo 8 64 64

