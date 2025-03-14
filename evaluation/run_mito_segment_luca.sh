#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH --job-name=fidi_mito_segment
#SBATCH --constraint 80gb

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/lustre-grete/usr/u12103/envs/synapse

python /user/freckmann15/u12103/synapse/segment_mitos_wichmann.py \
  -b /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/done_h5_s2 \
  -e /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/lucas_segment_test_s2_mito1_model_freshly_downloaded \
  -m /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/models/model.pt