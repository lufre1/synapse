import argparse
import os
from glob import glob
import torch_em
from tqdm import tqdm
from elf.io import open_file
import numpy as np
from synapse_net.inference.mitochondria import segment_mitochondria

# -m /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mito-domain-adapt-s2-sampler-ct5


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted/", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/new_mito_labels/", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/projects/nim00007/models/exports_for_cooper/mito_model_s2.pt")
    args = parser.parse_args()
    print(args.base_path)

    h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True))#, reverse=True)

    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": {"z": 40, "y": 512+128, "x": 512+128}, "halo": {"z": 8, "y": 128, "x": 128}}
    scale = 0.8

    for path in tqdm(h5_paths):
        with open_file(path, "r") as f:
            data = f["raw"][:]
            image = torch_em.transform.raw.standardize(data)
        output_path = os.path.join(args.export_path, "DA_scale_" + str(int(scale*10)) + "_" + os.path.basename(path))
        if os.path.exists(output_path):
            print("Skipping... output path exists", output_path)
            continue
        seg, pred = segment_mitochondria(
            image, args.model_path,
            scale=scale,
            tiling=tiling,
            return_predictions=True
            )
        with open_file(output_path, "w", ".h5") as f1:
            f1["labels/mitochondria"] = seg
            f1["pred"] = pred
            f1["raw"] = data
            print("Saved to", output_path)


if __name__ == "__main__":
    main()