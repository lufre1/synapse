import util
import data_classes
from config import *
import h5py
import argparse
import os
from glob import glob
import numpy as np
import torch_em


def _read_h5(path, key, scale_factor):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            image = f[key][:, ::scale_factor, ::scale_factor]
            print(f"{key} data shape after downsampling", image.shape)
            # if not key == "raw":
            #     print(np.unique(image))

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--pred_dir", type=str, default=SAVE_DIR, help="Path to the predictions' directory")    
    parser.add_argument("--file_path", type=str, default="", help="Path to a specific .h5 file to visualize")
    parser.add_argument("--pred_path", type=str, default="", help="Path to the prediction's file")
    parser.add_argument("--single_file_path", type=str, default="", help="Path singe file to visualize")
    parser.add_argument("--base_path", type=str, default="", help="Path to h5 directory with h5 keys 'raw' and 'labels/mitochondria' and 'labels/cristae'")
    
    
    args = parser.parse_args()
    raw_data_path = args.data_dir
    pred_dir = args.pred_dir
    pred_path = args.pred_path
    raw_file_path = args.file_path
    base_path = args.base_path
    
    scale_factor = 1
    
    if base_path:
        paths = glob(os.path.join(base_path, "**", "*.h5"), recursive=True)
        for path in paths:
            img = _read_h5(path, "raw", scale_factor)
            label = _read_h5(path, "labels/mitochondria", scale_factor)
            label2 = _read_h5(path, "labels/cristae", scale_factor)
            if label2 is not None:
                vis_data = {
                    "raw": img,
                    "label": label,
                    "label2": label2
                }
            else:
                vis_data = {
                    "raw": img,
                    "label": label
                }
            util.visualize_data_napari(vis_data)
            
    if args.single_file_path:
        image = None
        label = None
        image = _read_h5(args.single_file_path, "raw", scale_factor)
        label = _read_h5(args.single_file_path, "labels/mitochondria", scale_factor)
        label2 = _read_h5(args.single_file_path, "labels/cristae", scale_factor)
        mean = np.mean(image)
        image1 = torch_em.transform.raw.standardize(image, mean)
        if label2 is not None:
            vis_data = {
                "raw": image1,
                #"label": label,
                #"pred1": label2
            }
        else:
            vis_data = {
                "raw": image1,
                #"label": label
            }
        util.visualize_data_napari(vis_data)
        
    if args.file_path:
        data_paths = []
        data_paths.append(args.file_path)
    else:
        data_paths = glob(os.path.join(pred_dir, "**", "*.h5"), recursive=True)
    raw_data_paths = glob(os.path.join(raw_data_path, "**", "*.h5"), recursive=True)
    if raw_file_path:
        raw_data_paths = [raw_file_path]
    if pred_path:
        data_paths = [pred_path]
    for i, data_path in enumerate(data_paths):
        print(f"Visualizing {data_path}...")
        with h5py.File(data_path, "r") as f:
            print("Prediction shape:", f["prediction"].shape)
            pred = f["prediction"][:, :, ::int(scale_factor/2), ::int(scale_factor/2)]
            print("Prediction shape after downsampling:", pred.shape)
            threshold = .85
            pred_foreground = (pred[0, :, :, :] > threshold).astype(np.uint8)
            pred_boundaries = (pred[1, :, :, :] > (threshold-.6)).astype(np.uint8)
        # breakpoint()
        with h5py.File(raw_data_paths[i], "r") as f:
            if "raw" in f:
                print("Raw data shape", f["raw"].shape)
                image = f["raw"][:][:, ::scale_factor, ::scale_factor]
                print("Raw data shape after downsampling", image.shape)

            if "labels/mitochondria" in f:
                label = f["labels/mitochondria"][:][:, ::scale_factor, ::scale_factor]

        vis_data = {
            "raw": image,
            "label": label,
            "pred1": pred_foreground,
            "pred2": pred_boundaries
        }
        util.visualize_data_napari(vis_data)


if __name__ == "__main__":
    visualize()