import os
import torch
from tqdm import tqdm
import argparse
from elf.io import open_file
from glob import glob
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device, get_model_names
# from synapse_net.file_utils import read_ome_zarr


def compute_embeddings(path, output_path, model_type=None):
    
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    # all_models = get_model_names()
    if model_type is None:
        model_type = "vit_b_em_organelles"
    predictor = get_sam_model(model_type=model_type, device=device)
    try:
        if ".zarr" in path:
            # data, voxel_size = read_ome_zarr(path)
            raise NotImplementedError(
                f"Unsupported file type for '{path}'. "
                f"Expected a .ome.zarr/.zarr/.n5 directory or a .h5/.hdf5 file."
            )
        elif ".h5" in path:
            data = open_file(path)["raw"]
        else:
            raise NotImplementedError(
                f"Unsupported file type for '{path}'. "
                f"Expected a .ome.zarr/.zarr/.n5 directory or a .h5/.hdf5 file."
            )
    except Exception as e:
        # Wrap with context while preserving original traceback
        raise RuntimeError(f"Failed to load volume from '{path}': {type(e).__name__}: {e}") from e
        # precompute_image_embeddings(
        #         predictor, input_=data,
        #         save_path=out_filenpath, tile_shape=(512, 512), halo=(128, 128)                      
        #     )
    precompute_image_embeddings(
            predictor, input_=data,
            save_path=output_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/cryo-et-luca", help="Path to the root data directory")
    parser.add_argument("--output_path", "-o",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/tiled_embeddings_em", help="Path to the output data directory")
    parser.add_argument("--model_type", "-mt", default="vit_b_em_organelles", help="choose model type from (vit_b_em_organelles | vit_b | ...)")
    args = parser.parse_args()
    base_path = args.base_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.isdir(base_path) and base_path.endswith(".zarr"):
        print("Loading single zarr file:", base_path)
        paths = [base_path]
    else:
        paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    # exclude masks and other embeddings
    #paths[:] = [path for path in paths if "mask" not in path.lower() and "embeddings" not in path.lower()]
    for path in tqdm(paths):
        print("Precomputing embeddings for:", path)
        # output_path = path.replace(".zarr", "_embeddings")
        filename = os.path.basename(path)
        out_filepath = os.path.join(output_path, filename).replace(".h5", ".zarr")
        print("Output path:", out_filepath)
        if os.path.exists(out_filepath):
            print("Embeddings already precomputed for:", out_filepath)
            continue
        compute_embeddings(path, out_filepath, model_type=args.model_type)
        print("Precomuted embeddings and saved to:", output_path)
