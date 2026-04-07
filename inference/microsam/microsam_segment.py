import os
import torch
from tqdm import tqdm
import argparse
from elf.io import open_file
from glob import glob
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device, get_model_names
from micro_sam.inference import batched_tiled_inference, batched_inference
from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d
from skimage.measure import regionprops
import numpy as np
import tifffile
import yaml


def parse_args():
    # 1) tiny pre-parser: only reads -c/--config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", type=str, default=None, help="YAML config file")
    cfg_args, remaining = pre.parse_known_args()

    # 2) load defaults from YAML (if provided)
    cfg = {}
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # 3) main parser with YAML-provided defaults
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--input_path", "-i", type=str, default=cfg.get("input_path", None),
                        help="image path, can be None if embed path is provided")
    parser.add_argument("--checkpoint_path", "-cp", type=str, default=cfg.get("checkpoint_path", None), required=False)
    parser.add_argument("--model_type", "-m", type=str, default=cfg.get("model_type", "vit_b"),
                        help="The SAM model type to use")
    parser.add_argument("--embedding_path", "-ep", type=str, default=cfg.get("embedding_path", None))
    parser.add_argument("--export_path", "-e", type=str, default=cfg.get("export_path", None), required=False)
    parser.add_argument("--tile_shape", "-ts", nargs=2, type=int, default=cfg.get("tile_shape", None))
    parser.add_argument("--halo", "-ha", nargs=2, type=int, default=cfg.get("halo", None))
    parser.add_argument("--key", "-k", type=str, default=cfg.get("key", None))
    parser.add_argument("--segmentation_path", "-s", type=str, default=cfg.get("segmentation_path", None))
    parser.add_argument("--segmentation_key", "-segk", type=str, default=cfg.get("segmentation_key", None))

    args = parser.parse_args(remaining)

    # 4) enforce required args after merging YAML + CLI
    missing = [n for n in ("checkpoint_path", "export_path") if getattr(args, n) in (None, "")]
    if missing:
        parser.error(f"Missing required arguments (provide via CLI or YAML): {', '.join(missing)}")

    return args


def extract_prompts(path, key, min_size=0):
    """
    Extracts SAM point prompts from a purely 3D segmentation file.
    
    Args:
        path: Path to the data file.
        key: The dataset key within the file.
        min_size: Minimum area (in pixels) required to keep a prompt. 
                  Objects smaller than this will be ignored. Defaults to 0.
        
    Returns:
        prompts_by_slice: A dictionary where the key is the slice index (z) 
                          and the value is a tuple of (points_array, labels_array).
    """
    prompts_by_slice = {}
    with open_file(path) as f:
        seg = f[key]
        num_slices = seg.shape[0]
        for z in range(num_slices):
            # Load only the current 2D slice into memory to prevent OOM errors
            slice_seg = seg[z]
            
            # regionprops handles the unique IDs automatically and ignores background (0)
            regions = regionprops(slice_seg.astype(int))
            
            points = []
            labels = []
            
            for region in regions:
                if region.area >= min_size:
                    # skimage centroid returns (Y, X)
                    cy, cx = region.centroid
                    
                    # SAM requires coordinates in [X, Y] format, shape [1, 2]
                    points.append([[cx, cy]])
                    labels.append([1])  # 1 indicates a positive point prompt
            
            # Only add to the dictionary if the slice contains objects
            if points:
                prompts_by_slice[z] = (
                    np.array(points, dtype=np.float32),
                    np.array(labels, dtype=np.int32)
                )
                
    return prompts_by_slice
                

def main():
    args = parse_args()
    input_path = args.input_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_path = args.export_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.isdir(input_path) and input_path.endswith(".zarr"):
        print("Loading single zarr file:", input_path)
        paths = [input_path]
    else:
        paths = sorted(glob(os.path.join(input_path, "**", "*.h5"), recursive=True))
    predictor = get_sam_model(model_type=args.model_type, device=device, checkpoint_path=args.checkpoint_path)
    for path in tqdm(paths):
        print("Processing:", path)
        final_3d_segmentation = []
        if args.input_path is not None:
            with open_file(path) as f:
                raw = f[args.key]
                raw_shape = raw.shape
                num_slices = raw_shape[0]
        # start by loading the segmentation
                if args.segmentation_path is not None:
                    prompts_by_slice = extract_prompts(args.segmentation_path, args.segmentation_key, min_size=250)

                    for z in range(num_slices):
            
                        # Check if we actually have any prompts for this specific slice
                        if z not in prompts_by_slice:
                            # If no objects were in this slice's segmentation, skip inference
                            # (Append an empty mask or zeros to keep your 3D shape consistent)
                            final_3d_segmentation.append(np.zeros(raw_shape[1:], dtype=np.uint32))
                            continue 
                            
                        # 3. UNPACK THE DICTIONARY HERE
                        # This grabs the specific arrays your inference function demands
                        current_points, current_labels = prompts_by_slice[z]
                        
                        # 4. Pass the unpacked arrays directly to your inference function
                        if args.tile_shape and args.halo:
                            slice_masks = batched_tiled_inference(
                                image=raw,
                                predictor=predictor,
                                embedding_path=args.embedding_path,
                                batch_size=1,
                                points=current_points,         # Pass the unpacked points array
                                point_labels=current_labels,   # Pass the unpacked labels array
                                i=z,                           # Pass the slice index
                                optimize_memory=True,           # (or whatever your specific kwargs are)
                                tile_shape=args.tile_shape,
                                halo=args.halo,
                            )
                        else:
                            slice_masks = batched_inference(
                                image=raw,
                                predictor=predictor,
                                embedding_path=args.embedding_path,
                                batch_size=1,
                                points=current_points,         # Pass the unpacked points array
                                point_labels=current_labels,   # Pass the unpacked labels array
                                i=z,                           # Pass the slice index
                            )
                        
                        # Store the result
                        final_3d_segmentation.append(slice_masks)

                    # Stack everything back into a 3D volume when the loop finishes
                    final_3d_segmentation = np.stack(final_3d_segmentation, axis=0)
                    print("Merging 2D slices into coherent 3D objects...")
                    coherent_3d_segmentation = merge_instance_segmentation_3d(
                        slice_segmentation=final_3d_segmentation,
                        beta=0.5,             # 0.5 is default. Higher = more merging, Lower = more splitting
                        with_background=True,
                        gap_closing=1,        # Closes gaps if an axon briefly disappears for 1 slice
                        min_z_extent=2,       # Filters out noise: requires objects to exist on at least 2 slices
                        verbose=True
                    )

                    # export
                    output_file = os.path.join(
                        args.export_path, os.path.basename(path).replace(".zarr", "").replace(".h5", "") + "_sam.tif"
                    )
                    tifffile.imwrite(output_file, coherent_3d_segmentation.astype(np.uint16), compression="zlib")
                    print(f"Exported segmentation to {output_file}")


if __name__ == "__main__":
    main()