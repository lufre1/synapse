#!/bin/bash
#SBATCH --partition=grete:interactive
#SBATCH -G 1g.20gb:1
#SBATCH --time=0-6:00:00
#SBATCH --job-name=infer-cristae-tomo
#SBATCH -c 4
#SBATCH --mem 32G
## SBATCH --partition=grete:shared
## SBATCH -G A100:1


source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# ================ Define ALL parameters here ONCE ================
# cellmaps on volume em data
BLOCK_SHAPE="32 256 256"
DD="/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/"
EXPORT_PATH="/mnt/lustre-grete/usr/u12103/cristae_data/segmentations_cristae/test_split_updated"
FORCE_OVERRIDE=True
MODEL_PATH="/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-updated"



python /user/freckmann15/u12103/synapse/inference/segment_cristae.py \
  --base_path ${DD} \
  --export_path ${EXPORT_PATH} \
  --model_path ${MODEL_PATH} \
  -ts ${BLOCK_SHAPE} \