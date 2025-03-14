#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=mito-net


source /home/nimlufre/.bashrc
conda activate synapse

python /home/nimlufre/synapse/main.py \
  --experiment_name "mito-net32-bs2-ps32448-lr1e-3-isotropicscaling-labelthreshold-50-50" \
  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/training_blocks_v1/ \
  --n_iterations 100000 \
  --patch_shape 32 448 448 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --feature_size 32 \
 # --without_rois True


# /scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/ 

# srun /home/nimlufre/miniforge3/envs/synapse/bin/python /home/nimlufre/synapse/main.py \
#   --experiment_name mito-net-bs1-ps-48 \
#   --patch_shape 48 384 384 \ 448 also possible as well as 480
#   --batch_size 1 \
# 
# #SBATCH -c 16
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-gpu 16
# #SBATCH --constraint=80gb
# #SBATCH --qos=14d
# #SBATCH --mem-per-gpu 128G
# #SBATCH --gres=gpu:A100:1