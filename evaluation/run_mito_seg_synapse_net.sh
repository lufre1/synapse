#!/bin/bash

input_files=(
    "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5"
    "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/WT21_eb3_model2.h5"
    "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M10_eb9_model.h5"
    "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/KO9_eb4_model.h5"
    "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M7_eb11_model.h5"
    "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_PS_25_rec_2Kb1dawbp_crop_downscaled.h5"
)

output_dir="mito_thick_out"
checkpoint="/user/freckmann15/u12103/synapse/models/mito-v3"
data_ext=".h5"
mode="mitochondria"

synapse_net.run_segmentation \
    -i '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5' \
    -o mito_thick_out \
    -c /user/freckmann15/u12103/synapse/models/mito-v3 \
    --data_ext .h5 \
    -m mitochondria

for input_file in "${input_files[@]}"; do
    echo "Processing: $input_file"
    synapse_net.run_segmentation \
        -i "$input_file" \
        -o "$output_dir" \
        -c "$checkpoint" \
        --data_ext "$data_ext" \
        -m "$mode"
done