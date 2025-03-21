import os
import torch
from tqdm import tqdm
import argparse
from elf.io import open_file
from glob import glob
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device, get_model_names
from synapse_net.file_utils import read_ome_zarr


def compute_embeddings(path, output_path):
    filename = os.path.basename(path)
    out_filenpath = os.path.join(output_path, filename)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    # all_models = get_model_names()
    model_type = "vit_b_em_organelles"
    predictor = get_sam_model(model_type=model_type, device=device)
    data, voxel_size = read_ome_zarr(path)
    precompute_image_embeddings(
            predictor, input_=data,
            save_path=out_filenpath, tile_shape=(512, 512), halo=(128, 128)                      
        )
    # with open_file(path, "r") as f:
    #     raw = f["raw"][:]
    #     precompute_image_embeddings(
    #         predictor, input_=raw,
    #         save_path=out_filenpath, tile_shape=(512, 512), halo=(128, 128)                      
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/cryo-et-luca", help="Path to the root data directory")
    parser.add_argument("--output_path", "-o",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/tiled_embeddings_em", help="Path to the output data directory")
    args = parser.parse_args()
    base_path = args.base_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    paths = sorted(glob(os.path.join(base_path, "**", "*.zarr"), recursive=True))
    paths[:] = [path for path in paths if "mask" not in path.lower()]  # exclude masks
    for path in tqdm(paths):
        output_path = path.replace(".zarr", "_embeddings.zarr")
        compute_embeddings(path, output_path)
        print("Precomuted embeddings and saved to:", output_path)
