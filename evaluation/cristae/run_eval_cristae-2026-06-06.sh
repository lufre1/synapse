#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval-cristae-net32-2026-06-01
#SBATCH -c 8
#SBATCH --mem 64G

MODEL="cristae-net32-lr1e-4-bs8-ps32x256x256-2026-06-04"
MODEL_PATH="/mnt/lustre-grete/usr/u12103/cristae/checkpoints/${MODEL}"
EXPORT_PATH="/mnt/lustre-grete/usr/u12103/cristae/test_segmentations/${MODEL}"
EVAL_CSV="${EXPORT_PATH}/cristae_eval_results.csv"

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

echo "=== Step 1: Segment test split ==="

python - << PYEOF
from synapse.cristae.segment import run_cristae_segmentation

test_paths = [
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT22_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT40_eb10_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M5_eb1_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36859_J1_66K_TS_PS_03_rec_2kb1dawbp_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_SC_22_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/KO8_eb4_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/2026-05-26-dataset/2026-05-26_corrected_combined/Otof_AVCN07_455L_KO_M.Stim_B3_2_35933_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M8_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT20_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M1_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M2_eb5_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/Otof_AVCN03_429A_WT_M.Stim_D3_4model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_R01A_SC_01_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_syn4_model2_combined.h5",
]

run_cristae_segmentation(
    h5_paths=test_paths,
    model_path="$MODEL_PATH",
    export_path="$EXPORT_PATH",
    tile_shape=(32, 512, 512),
)
PYEOF

echo "=== Step 2: Evaluate ==="

# The segment step copies labels/cristae from each input file into the output,
# so we point both labels and segmentations at the same export directory.
python /user/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py \
  -l "${EXPORT_PATH}" \
  -s "${EXPORT_PATH}" \
  -k labels/cristae \
  -sk seg \
  -o "${EVAL_CSV}"

echo "Done. Results: ${EVAL_CSV}"
