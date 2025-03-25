#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --mem 128G
#SBATCH --job-name=domain-adapt-mito
#SBATCH --constraint 80gb


source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/train_mito_domain_adaptation.py \
  --experiment_name "domain-adapt-cryoet-ps32512-bs1" \
  --data_dir /scratch-grete/projects/nim00007/cryo-et-luca/ \
  --checkpoint_path "/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitotomo-net32-bs2-ps32512-lr1e-4-downscaled" \
  --confidence_threshold 0.5 \
  --n_iterations 30000 \
  --batch_size 1 \
  --patch_shape 32 512 512 
    # 22000 should be approx 4 epochs

# mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held_s2/xx