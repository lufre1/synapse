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
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/trimmed_all", help="Path to the root data directory")
    parser.add_argument("--label_path", "-lp",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/new_mito_labels", help="Path to the root data directory")
    parser.add_argument("--embeddings_path", "-ep",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/tiled_embeddings_em", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/home/freckmann15/data/mitochondria/wichmann/test/", help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--overlap_threshold", "-ot", type=float, default=0.5, help="Overlap threshold for filtering")
    parser.add_argument("--output_path", "-o",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/manual_and_microsam_annotations", help="Path to the output data directory")
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

    h5_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    h5_label_paths = sorted(glob(os.path.join(label_path, "**", "*.h5"), recursive=True))
    embeddings_paths = sorted(glob(os.path.join(embeddings_path, "**", "*.h5"), recursive=True))
    existing_output_files = sorted(glob(os.path.join(output_path, "**", "*.tif"), recursive=True))
    
    
    for h5_path in tqdm(h5_paths):
        #label_path = find_trimmed_and_new_labels_pair(h5_path, h5_label_paths, type="label")
        embeddings_path = find_trimmed_and_new_labels_pair(h5_path, embeddings_paths, type="embedding")
        # if label_path is None:
        #     continue
        
        current_filename = os.path.basename(h5_path)
        
        # new_files_list = [
        #     "Otof_AVCN03_429D_WT_Rest_H5_1_35461_model", "Otof_AVCN03_429D_WT_Rest_H5_3_35461_model", "Otof_AVCN03_429D_WT_Rest_H5_4_35461_model",
        #     "WT40_eb10_model", "WT41_eb4_model"
        # ]
        
        # for new_file in new_files_list:
        #     print("new file and current file:", new_file, current_filename)
        #     if new_file in current_filename:
        #         skip = False
        #         breakpoint()
        #         continue
        #     else:
        #         skip = True
                
        skip = False
        for file in existing_output_files:
            if current_filename.replace(".h5", "") in file:
                print("output file already exists:", file)
                skip = True
                continue
        if skip:
            continue
        keys = get_all_keys_from_h5(h5_path)
        data = {}
        for key in keys:
            data[key] = read_h5(h5_path, key, scale_factor)
        #new_labels = read_h5(label_path, "labels/mitochondria", scale_factor)
        
        #data["labels/mitochondria"] = (new_labels + (data["labels/mitochondria"] > 0).astype(np.uint8) > 0).astype(np.uint8)
        output_path = os.path.join(export_path, os.path.basename(h5_path))
        print("all paths:", h5_path, label_path, embeddings_path)
        orig_labels = data["labels/mitochondria"]

        # orig_labels, new_labels = match_z_slices(data["labels/mitochondria"], new_labels)
        orig_labels = relabel_sequential(orig_labels)[0]
        # new_labels = relabel_sequential(new_labels)[0]
        # overlap, _ = label_overlap(new_labels, data["labels/mitochondria"])
        # iou = intersection_over_union(overlap)
        # # print(label_overlap(data["labels/mitochondria"], new_labels))
        # # print("intersection over union:", iou)
        # seg_ids = np.unique(new_labels)
        # filtered_ids = []
        # for seg_id in seg_ids:
        #     max_overlap = iou[seg_id, :].max()
        #     if max_overlap > overlap_threshold:
        #         filtered_ids.append(seg_id)
        # additional_objects = new_labels.copy()
        # additional_objects[np.isin(new_labels, filtered_ids)] = 0
        # additional_objects = relabel_sequential(additional_objects)[0]
        if tile_shape is None:
            annotator_3d(model_type=model_type, image=data["raw"], segmentation_result=orig_labels, embedding_path=embeddings_path)
        else:
            annotator_3d(model_type=model_type, image=data["raw"], segmentation_result=orig_labels, embedding_path=embeddings_path,
                         tile_shape=tile_shape, halo=halo)
        
        # if os.path.exists(output_path):
        #     print("output path already exists:", output_path)
        #     continue
        # else:
        #     export_to_h5(data, output_path)
        
        # v = napari.Viewer()
        # v.add_image(data["raw"])
        # v.add_labels(label(orig_labels), name="orig")
        # v.add_labels(new_labels, name="new_labels")
        # v.add_labels(additional_objects, name="additional_objects")
        # napari.run()


if __name__ == "__main__":
    main()