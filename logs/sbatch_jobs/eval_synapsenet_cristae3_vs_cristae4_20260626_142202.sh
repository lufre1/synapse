#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1-12:00:00
#SBATCH --nodes 1
#SBATCH -p grete:shared
#SBATCH -G A100:1
source ~/.bashrc
micromamba activate /mnt/vast-nhr/home/freckmann15/u12103/micromamba/envs/synapse
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/make_test_split_dir.py --out /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/test_gt && \
set -euo pipefail && synapse_net.run_segmentation -i /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/test_gt -o /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae3 -m cristae3 --data_ext .h5 --segmentation_key cristae --scale 1.208 --tile_shape 32 512 512 --halo 8 128 128 --force -v && \
set -euo pipefail && python -c "import glob,os; [os.replace(f, f.replace('_prediction.h5','.h5')) for f in glob.glob('/mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae3/*_prediction.h5')]" && \
set -euo pipefail && synapse_net.run_segmentation -i /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/test_gt -o /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae4 -m cristae4 --data_ext .h5 --segmentation_key cristae --scale 1.0 --tile_shape 32 512 512 --halo 8 128 128 --force -v && \
set -euo pipefail && python -c "import glob,os; [os.replace(f, f.replace('_prediction.h5','.h5')) for f in glob.glob('/mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae4/*_prediction.h5')]" && \
set -euo pipefail && mkdir -p /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae3 /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae4 && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py -l /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/test_gt -k labels/cristae -le .h5 -s /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae3 -sk cristae -se .h5 -o /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae3 -hd && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/evaluate_cristae.py -l /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/test_gt -k labels/cristae -le .h5 -s /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/seg_cristae4 -sk cristae -se .h5 -o /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae4 -hd && \
set -euo pipefail && python /mnt/vast-nhr/home/freckmann15/u12103/synapse/evaluation/cristae/compare_eval_csvs.py -c cristae3 /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae3/cristae_eval_results.csv -c cristae4 /mnt/lustre-grete/usr/u12103/cristae/test_segmentations/synapsenet_cli/eval_cristae4/cristae_eval_results.csv

echo 'Done.'
