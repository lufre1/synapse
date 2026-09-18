#!/usr/bin/env bash
# Export Leonie Schadt's IMOD annotations (optic nerve FIB-SEM, Plp KO vs WT) to zarr labels.
# Run inside `mamba activate synapse`. Latest model per animal only.
set -euo pipefail
R=/home/freckmann15/data/volume-em
M="$R/mod files for volume em/Constantin Pape"
S="$(dirname "$0")/imod_to_zarr.py"

python "$S" --selftest

# 4010 (WT) has raw + auto segmentation locally -> orientation smoke test / manual ground truth
python "$S" "$M/4010 wt/2019-07-31_paper-4010-wt.mod" -o "$R/WT/4010/4010_imod.zarr"
# 4007 (KO) counterpart of the 4010 paper model. Model has 1448 z-slices, raw has 1447;
# exported at model size, so it is one slice taller than 4007_raw_v3.zarr.
python "$S" "$M/4007 Plp/2019-08-01_PLP-paper_4007.mod" -o "$R/KO/4007/4007_imod.zarr"

# animals WITHOUT local raw data
python "$S" "$M/4005 Plp/2019-06-09_PLP_4005.mod" -o "$R/KO/4005/4005_imod.zarr"
python "$S" "$M/4005 Plp/Left_Axon_Final.mod=Axon Left" "$M/4005 Plp/Right_Axon_Final.mod=Axon Right" \
            "$M/4005 Plp/Left_Myelin_Final.mod=Myelin Left" "$M/4005 Plp/Right_Myelin_Final.mod=Myelin Right" \
            -o "$R/KO/4005/4005_crop_imod.zarr"   # 4216x4920x336 crop, offset vs main volume unknown
python "$S" "$M/4009 wt/2019-06-12_wt-4009.mod" -o "$R/WT/4009/4009_imod.zarr"
python "$S" "$M/4009 wt/2020-04-08_Plp-4009-wt-axons.mod" -o "$R/WT/4009/4009_imod.zarr" --suffix _2020
python "$S" "$M/4015 Plp/4015_2019-05-09_model_mitos.mod" -o "$R/KO/4015/4015_imod.zarr"   # z-scale 2 -> 10x5x5 nm
python "$S" "$M/4016 wt/4016_2019-06-11_model_mitos.mod" -o "$R/WT/4016/4016_imod.zarr"

# skipped on purpose:
#   4005 Plp/2020-01-16-axonal-swelling_4005.mod  869 MB, isosurface meshes only, 0 contours
#   4005 Plp/2020-01-21_model_4005.mod            1 unnamed object, purpose unknown
#   4016 wt/4026_2019-05-08_model_mitos.mod       typo copy of 4016, older than 2019-06-11
#   older versions of 4007/4010/4015/4016 models
