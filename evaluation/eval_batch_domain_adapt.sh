#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-00:10:00
#SBATCH --account=nim00007
#SBATCH --job-name=eval-domain-adapt
#SBATCH -c 8
#SBATCH --ntasks=1


exp_name="mito-domain-adapt-newsampler-ct5-ps32512-bs2-with_test"
#"mito-domain-adapt-s2-sampler-ct5-ps32512-with_test"
# "mitotomo-net32-bs2-ps64512-lr1e-4-all-mitos"
# "mitotomo-net32-bs2-ps64512-lr1e-4-all-mitos"
#"mito-domain-adapt-s2-sampler-ct5-with_test"
#"mito-domain-adapt-s2-sampler-ct5-ps32512-with_test"
#"mito-domain-adapt-s2-sampler"

source /home/nimlufre/.bashrc
conda activate synapse

python /user/freckmann15/u12103/synapse/test.py \
  --experiment_name $exp_name \
  --down_scale_factor 1 \
  --checkpoint_path /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/$exp_name \
  --patch_shape 32 512 512 \
  --file_path /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Automatische_Segmentierung_Dataset_Validierung/1Otof_AVCN07_451A_WT_Rest_B3_10_35932.h5
  # --data_dir /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/
  # --file_path /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Young_KO_MStim/1Otof_AVCN03_439G_KO_M.Stim_M3_3.h5
  
  # /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_MF_18_rec_2Kb1dawbp_crop_downscaled.h5

  # /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/36859_J1_66K_TS_CA3_PS_52_rec_2Kb1dawbp_crop.h5
# /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/36859_J1_66K_TS_CA3_PS_23_rec_2Kb1dawbp_crop.h5
#  --data_dir /scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/training_blocks_v1/ 

### downscaled test data
# /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_MF_18_rec_2Kb1dawbp_crop_downscaled.h5
# /scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/3.2_downscaled.h5