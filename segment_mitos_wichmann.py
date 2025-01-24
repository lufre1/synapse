import argparse
import os
from glob import glob
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
from synapse_net.inference.mitochondria import segment_mitochondria

# -m /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mito-domain-adapt-s2-sampler-ct5


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted/", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/test/", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/projects/nim00007/models/exports_for_cooper/mito_model_s2.pt")
    args = parser.parse_args()
    print(args.base_path)
    # tile_shape
    ts = {
        "z": 48,
        "y": 512,
        "x": 512
        }
    halo = {
        "z": int(ts["z"] * 0.1),
        "y": int(ts["y"] * 0.25),
        "x": int(ts["x"] * 0.25)
        }
    # halo = {'z': 12, 'y': 128, 'x': 128}
    ts = {'z': ts["z"]+2*halo["z"], 'y': ts["y"]+2*halo["y"], 'x': ts["x"]+2*halo["x"]}
    h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True), reverse=True)

    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": ts, "halo": halo} # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = 1

    for path in tqdm(h5_paths):
        skip = True
        if "KO8_eb2_model" in path or "KO9_eb6_model" in path or "M7_eb2_model" in path:
            # breakpoint()
            skip = False
        if skip:
            continue
        print("opening file", path)
        output_path = os.path.join(args.export_path, "_" + os.path.basename(args.model_path) + "_sd6_" + os.path.basename(path))
        if os.path.exists(output_path):
            print("Skipping... output path exists", output_path)
            continue
        with open_file(path, "r") as f:
            data = f["raw"][:]
            # mean = np.mean(data)
            # valid_min, valid_max = -5, 5
            # valid_mask = (data >= valid_min) & (data <= valid_max)
            # data[~valid_mask] = mean
            # min_val = np.min(valid_data)
            # max_val = np.max(valid_data)
            # data = data[valid_mask] = 2 * (valid_data - min_val) / (max_val - min_val) - 1
            image = torch_em.transform.raw.standardize(data)
        

        seg, pred = segment_mitochondria(
            image, args.model_path,
            scale=scale,
            tiling=tiling,
            return_predictions=True,
            min_size=50000*3,
            seed_distance=6,  # default 6
            ws_block_shape=(128, 256, 256),
            ws_halo=(64, 128, 128),
            )
        with open_file(output_path, "w", ".h5") as f1:
            f1["labels/mitochondria"] = seg
            f1["pred"] = pred
            f1["raw"] = data
            print("Saved to", output_path)


if __name__ == "__main__":
    main()