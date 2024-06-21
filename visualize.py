import util
import data_classes
from config import *
import h5py
import argparse
import os
from glob import glob
import numpy as np


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--pred_dir", type=str, default=SAVE_DIR, help="Path to the predictions' directory")    
    parser.add_argument("--file_path", type=str, default="", help="Path to a specific .h5 file to visualize")
    parser.add_argument("--pred_path", type=str, default="", help="Path to the prediction's file")
    
    args = parser.parse_args()
    raw_data_path = args.data_dir
    pred_dir = args.pred_dir
    pred_path = args.pred_path
    raw_file_path = args.file_path

    scale_factor = 3

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
            pred = f["prediction"][:, :, ::scale_factor, ::scale_factor]
            print("Prediction shape after downsampling:", pred.shape)
            threshold = .75
            pred_foreground = (pred[0, :, :, :] > threshold).astype(np.uint8)
            pred_boundaries = (pred[1, :, :, :] > (threshold-.2)).astype(np.uint8)
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
