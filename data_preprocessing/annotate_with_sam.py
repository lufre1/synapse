import os
from glob import glob
import napari
import numpy as np
import skimage
from tqdm import tqdm
import argparse
from synapse_net.file_utils import read_ome_zarr
from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d


def main(args):
    base_path = args.base_path
    embeddings_base_path = args.embedding_path
    model_type = args.model_type
    tile_shape = args.tile_shape
    halo = args.halo
    with_tiling = False
    fe = args.file_extension
    ee = args.embedding_extension

    paths = sorted(glob(os.path.join(base_path, "**", f"*{fe}"), recursive=True))
    embeddings_paths = sorted(glob(os.path.join(embeddings_base_path, "**", f"*{ee}"), recursive=True))
    
    ## just for this setup
    ## find substring in path that also occurs in embeddings path
    ## my substring is always between "block" and ".h5"
    new_paths = []
    for emb in embeddings_paths:
        substring = emb.split("block")[1].split(ee)[0]
        for path in paths:
            if path.find(substring) != -1:
                new_paths.append(path)
    paths = new_paths
    print("paths", len(paths))
    print("embeddings paths", len(embeddings_paths))

    for path, embeddings_path in tqdm(zip(paths, embeddings_paths), desc="Annotating"):
        # load data
        image = open_file(path).get(args.raw_key)[...]
        extra_segmentation = open_file(path).get("labels/mitochondria")[...]
        # v = napari.Viewer()
        # v.add_image(image)
        # napari.run()
        # embeddings_path = embeddings_path + "/features"

        print("raw path filename", os.path.basename(path))
        print("embeddings path", embeddings_path)
        if with_tiling:
            v = annotator_3d(
                model_type=model_type, image=image,  # segmentation_result=label_data,
                embedding_path=embeddings_path,  # prompts=prompts,
                tile_shape=tile_shape, halo=halo,
                return_viewer=True,
            )
        else:
            v = annotator_3d(
                model_type=model_type, image=image,  # segmentation_result=outer,
                embedding_path=embeddings_path,  # prompts=prompts
                return_viewer=True,
            )
        v.add_labels(extra_segmentation)
        napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str,
                        default="/home/freckmann15/data/cryo-et/upload_CZCDP-10010/11971",
                        help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".zarr", help="File extension to read data")
    parser.add_argument("--raw_key", "-rk",  type=str, default="raw", help="Raw key to be used for loading raw dataset")
    parser.add_argument("--embedding_path", "-e",  type=str, default=None, help="Path to the embedding directory")
    parser.add_argument("--embedding_extension", "-ee",  type=str, default=".zarr", help="File extension to read data")
    parser.add_argument("--tile_shape", "-ts",  type=tuple, default=(512, 512), help="Tile shape")
    parser.add_argument("--halo", "-halo",  type=tuple, default=(128, 128), help="Halo")
    parser.add_argument("--with_tiling", "-wt",  action='store_true', default=False, help="Use to train with ROIs (manually set ROI in script!!)")
    parser.add_argument("--model_type", "-m",  type=str, default="vit_b_em_organelles", help="Model type")
    args = parser.parse_args()
    main(args)
