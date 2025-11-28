#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=precompute-embeddings-sam
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -C inet
##SBATCH --partition=grete:shared
##SBATCH -G A100:1

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/sam

# ================ Define ALL parameters here ONCE ================

DD="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/final_h5/"
EXPORT_PATH="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-mitopaper/4007_split/train_split_embeddings"
MODEL_TYPE="vit_b_em_organelles"

python /user/freckmann15/u12103/synapse/data_preprocessing/precompute_embeddings_with_sam.py \
  -b ${DD} \
  -o ${EXPORT_PATH} \
  -mt ${MODEL_TYPE} \