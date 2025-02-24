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

python /user/freckmann15/u12103/synapse/segment_mitos_wichmann.py \
  -b /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250212_test_I_h5_s2/ \
  -e /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_test_segs/ \