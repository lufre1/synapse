#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=1-16:00:00
#SBATCH --job-name=cristae-masked-ab
#SBATCH -c 8
#SBATCH --mem 64G

# From-scratch run with MaskedDiceLoss to establish a clean baseline.
# Tests whether the 06-04 regression was caused by the checkpoint-precedence bug.

PATCH_SHAPE="32 256 256"
BS=8
LR=1e-4
ITER=125000
read -r PZ PY PX <<< "$PATCH_SHAPE"
EXPNAME="cristae-net32-lr${LR}-bs${BS}-ps${PZ}x${PY}x${PX}-masked-scratch"

DD1="/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2"
DD2="/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae"
DD3="/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

python /user/freckmann15/u12103/synapse/training/cristae/train_cristae.py \
  --experiment_name ${EXPNAME} \
  --patch_shape ${PATCH_SHAPE} \
  --n_iterations ${ITER} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --data_dir ${DD1} \
  --data_dir2 ${DD2} \
  --data_dir3 ${DD3} \
  --ignore_state_value 2 \
  --state_channel 1 \
  --loss_variant new \
  --test_split synapsenetv1-testsplit \
  --early_stopping 15 \
  --seed 42 \
  --normalize
