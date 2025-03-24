import os
from glob import glob
import numpy as np
import skimage
from tqdm import tqdm
import argparse
from synapse_net.file_utils import read_ome_zarr
from micro_sam.sam_annotator import annotator_3d


def _to_bbox(seg):
    coords = np.where(seg)

    if len(coords[0]) == 0:  # Handle empty segmentation
        return None

    if seg.ndim == 3:
        zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
        zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()

        # Define the 8 corners of the 3D bounding box
        return [
            [zmin, ymin, xmin],
            [zmin, ymax, xmin],
            [zmin, ymax, xmax],
            [zmin, ymin, xmax],
            [zmax, ymin, xmin],
            [zmax, ymax, xmin],
            [zmax, ymax, xmax],
            [zmax, ymin, xmax],
        ]
    else:  # 2D case
        ymin, xmin = coords[0].min(), coords[1].min()
        ymax, xmax = coords[0].max(), coords[1].max()
        return [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]


def mask_to_mesh(seg):
    """Convert 3D segmentation mask to a mesh using Marching Cubes."""
    verts, faces, _, _ = skimage.measure.marching_cubes(seg, level=0.5)
    return {
        "verts": verts,
        "faces": faces
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str,
                        default="/home/freckmann15/data/cryo-et/upload_CZCDP-10010/11971",
                        help="Path to the root data directory")
    parser.add_argument("--tile_shape", "-ts",  type=tuple, default=(512, 512), help="Tile shape")
    parser.add_argument("--halo", "-halo",  type=tuple, default=(128, 128), help="Halo")
    parser.add_argument("--model_type", "-m",  type=str, default="vit_b_em_organelles", help="Model type")
    args = parser.parse_args()
    base_path = args.base_path
    model_type = args.model_type
    tile_shape = args.tile_shape
    halo = args.halo
    with_tiling = False

    paths = sorted(glob(os.path.join(base_path, "**", "*.zarr"), recursive=True))
    # filter for only segmentations via mask
    outer_segmentation_paths = [path for path in paths if "mask" in path.lower() and "inner" not in path.lower()]
    inner_segmentation_paths = [path for path in paths if "mask" in path.lower() and "outer" not in path.lower()]
    raw_paths = [path for path in paths if "mask" not in path.lower()]
    embeddings_raw_path = [path for path in paths if "embedding" in path.lower() and "tiling" not in path.lower()]
    embeddings_path = embeddings_raw_path
    # embeddings_path = sorted(glob(os.path.join(embeddings_raw_path[0], "**", "*.zarr"), recursive=True))

    for path, outer_path, inner_path, embeddings_path in tqdm(
            zip(raw_paths, outer_segmentation_paths, inner_segmentation_paths, embeddings_path), desc="Annotating"):
        # load data
        data, _ = read_ome_zarr(path)
        # load labels
        outer, _ = read_ome_zarr(outer_path)
        outer = np.flip(outer, axis=1) if outer.ndim == 3 else np.flip(outer, axis=0)
        inner, _ = read_ome_zarr(inner_path)
        inner = np.flip(inner, axis=1) if inner.ndim == 3 else np.flip(inner, axis=0)
        print("raw path", path)
        print("segmentation path", outer_path)
        print("embeddings path", embeddings_path)
        # bbox = _to_bbox(outer)
        bbox = _to_bbox(outer)
        # mesh = mask_to_mesh(outer)
        print("bbox", bbox)
        prompts = {
            "prompts": bbox,
            # "mesh": mesh
        }
        if with_tiling:
            annotator_3d(model_type=model_type, image=data,  # segmentation_result=label_data,
                         embedding_path=embeddings_path, prompts=prompts,
                         tile_shape=tile_shape, halo=halo)
        else:
            annotator_3d(model_type=model_type, image=data,  # segmentation_result=outer,
                         embedding_path=embeddings_path, prompts=prompts)


if __name__ == "__main__":
    main()
