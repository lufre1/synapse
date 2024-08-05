#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=mito-net32


source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/main.py \
  --experiment_name "mitotomo-net32-bs2-ps64512-lr1e-4-fididata" \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi \
  --n_iterations 100000 \
  --patch_shape 64 512 512 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --feature_size 32 \
  --without_rois 1