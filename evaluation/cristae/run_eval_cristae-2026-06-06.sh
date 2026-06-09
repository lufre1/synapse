#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=0-3:00:00
#SBATCH --job-name=eval-cristae-net32-2026-06-04
#SBATCH -c 8
#SBATCH --mem 64G

source /user/freckmann15/u12103/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse

# Fail loudly: stop on first error, undefined variable, or failed pipe stage.
# Placed AFTER env activation, since conda/micromamba init scripts reference unset vars.
set -euo pipefail

REPO=/user/freckmann15/u12103/synapse
CONFIG="${REPO}/evaluation/cristae/eval_cristae-net32-lr1e-4-bs8-ps32x256x256-2026-06-04.yaml"

echo "=== Step 1: Segment test split ==="
python "${REPO}/inference/cristae/segment_cristae.py" -c "${CONFIG}"

echo "=== Step 2: Evaluate ==="
# segment writes labels/cristae and seg into export_path; evaluate_cristae.py
# defaults both labels_path and segmentations_path to export_path from the config.
python "${REPO}/evaluation/cristae/evaluate_cristae.py" -c "${CONFIG}"

echo "Done."
