import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage import measure
import argparse
# import numpy as np
import mrcfile
from synapse.util import get_data_metadata
import synapse.io.util as io 
from synapse.h5_util import read_h5, get_all_keys_from_h5
import napari
from elf.io import open_file
# from elf.parallel import label
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.morphology import label
from micro_sam.sam_annotator import annotator_3d


def find_trimmed_and_new_labels_pair(t_path, nl_paths, type):
    t_name = os.path.basename(t_path)
    for nl_path in nl_paths:
        nl_name = os.path.basename(nl_path)
        if t_name in nl_name:
            
            print(f"Found new {type} file for trimmed file:\n", nl_path)
            return nl_path
    print(f"Could not find new {type} file for trimmed file", t_name)
    return None


def match_z_slices(orig_labels, new_labels):
    # Get the shapes
    orig_shape = orig_labels.shape
    new_shape = new_labels.shape

    # Initialize updated labels as the original ones
    updated_orig_labels = orig_labels
    updated_new_labels = new_labels

    if orig_shape[0] < new_shape[0]:
        z_diff = new_shape[0] - orig_shape[0]
        print(f"Adding {z_diff} z-slice(s) to orig_labels.")
        new_slice = np.zeros(orig_labels.shape[1:], dtype=orig_labels.dtype)  # Fill with zeros
        new_slices = np.stack([new_slice] * z_diff, axis=0)
        updated_orig_labels = np.concatenate([orig_labels, new_slices], axis=0)
    elif new_shape[0] < orig_shape[0]:
        z_diff = orig_shape[0] - new_shape[0]
        print(f"Adding {z_diff} z-slice(s) to new_labels.")
        new_slice = np.zeros(new_labels.shape[1:], dtype=new_labels.dtype)  # Fill with zeros
        new_slices = np.stack([new_slice] * z_diff, axis=0)
        updated_new_labels = np.concatenate([new_labels, new_slices], axis=0)
    else:
        print("Both arrays already have the same number of z-slices.")

    return updated_orig_labels, updated_new_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/embl/cutout_1/images/ome-zarr/raw.ome.zarr", help="Path to the root data directory")
    parser.add_argument("--key", "-k", type=str, default=None, help="If given, only load key and raw from file to visualize")
    parser.add_argument("--label_path", "-lp",  type=str, default="/home/freckmann15/data/embl/cutout_1/images/ome-zarr/correction_v1.ome.zarr", help="Path to the root data directory")
    parser.add_argument("--embeddings_path", "-ep",  type=str, default="/home/freckmann15/data/embl/cutout_1/images/ome-zarr/raw.ome_embeddings/raw.ome.zarr", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/home/freckmann15/data/mitochondria/wichmann/test/", help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--overlap_threshold", "-ot", type=float, default=0.5, help="Overlap threshold for filtering")
    parser.add_argument("--output_path", "-o",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/manual_and_microsam_annotations", help="Path to the output data directory")
    parser.add_argument("--return_viewer", "-rv", action="store_true", default=False, help="Return viewer")
    args = parser.parse_args()
    base_path = args.base_path
    label_path = args.label_path
    export_path = args.export_path
    scale_factor = args.scale_factor
    embeddings_path = args.embeddings_path
    overlap_threshold = args.overlap_threshold
    output_path = args.output_path
    tile_shape, halo = None, None
    model_type = "vit_b_em_organelles"
    if "tile" in embeddings_path:
        tile_shape, halo = (512, 512), (128, 128)

    # h5_paths = sorted(glob(os.path.join(base_path, "**", "*.zarr"), recursive=True))
    paths = io.load_file_paths(base_path, ".zarr")
    label_paths = io.load_file_paths(label_path, "*.zarr")
    # embeddings_paths = sorted(glob(os.path.join(embeddings_path, "**", "*.h5"), recursive=True))
    existing_output_files = sorted(glob(os.path.join(output_path, "**", "*.tif"), recursive=True))

    for h5_path in tqdm(paths):

        # embeddings_path = find_trimmed_and_new_labels_pair(h5_path, embeddings_paths, type="embedding")

        current_filename = os.path.basename(h5_path)
   
        skip = False
        for file in existing_output_files:
            if current_filename.replace(".h5", "") in file:
                print("output file already exists:", file)
                skip = True
                continue
        if skip:
            continue
        # keys = get_all_keys_from_h5(h5_path)
        data = {}
        # for key in keys:
        #     data[key] = read_h5(h5_path, key, scale_factor)
        raw = io.load_data_from_file(h5_path, scale=scale_factor)

        for k, v in raw.items():
            data["raw"] = v
        
        labels = io.load_data_from_file(label_path, scale=scale_factor)
        for k, v in labels.items():
            data["labels"] = v

        # data = read_h5(h5_path, "raw", scale_factor)
        # label = read_h5(h5_path, "labels/mitochondria", scale_factor)

        output_path = os.path.join(export_path, os.path.basename(h5_path))
        print("all paths:", h5_path, label_path, embeddings_path)
        # orig_labels = data["labels/mitochondria"]

        # orig_labels, new_labels = match_z_slices(data["labels/mitochondria"], new_labels)
        # orig_labels = relabel_sequential(orig_labels)[0]

        if tile_shape is None:
            viewer = annotator_3d(model_type=model_type, image=data["raw"], segmentation_result=None,
                         embedding_path=embeddings_path, return_viewer=args.return_viewer)
        else:
            annotator_3d(model_type=model_type, image=data["raw"], segmentation_result=orig_labels, embedding_path=embeddings_path,
                         tile_shape=tile_shape, halo=halo)
        if args.return_viewer:
            viewer.add_labels(data["labels"], name="labels")
            print(viewer.layers)
            napari.run()


if __name__ == "__main__":
    main()