import util
import data_classes
from config import *
import h5py
import argparse
import os
from glob import glob


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--pred_dir", type=str, default=SAVE_DIR, help="Path to the predictions' directory")    
    parser.add_argument("--file_path", type=str, default="", help="Path to a specific .h5 file to visualize")
    
    args = parser.parse_args()
    raw_data_path = args.data_dir
    pred_dir = args.pred_dir
    if args.file_path:
        data_paths = []
        data_paths[0] = args.file_path
    else:
        data_paths = glob(os.path.join(pred_dir, "**", "*.h5"), recursive=True)
    raw_data_paths = glob(os.path.join(raw_data_path, "**", "*.h5"), recursive=True)
    for i, data_path in enumerate(data_paths):
        print(f"Visualizing {data_path}...")
        with h5py.File(data_path, "r") as f:
            pred = f["prediction"]
            #print(pred.shape)
            pred_foreground = pred[0, :, :, :]
            pred_boundaries = pred[1, :, :, :]
        # breakpoint()
        with h5py.File(raw_data_paths[i], "r") as f:
            if "raw" in f:
                image = f["raw"][:]
            if "labels/mitochondria" in f:
                label = f["labels/mitochondria"][:]

        vis_data = {
            "raw": image,
            "label": label,
            "pred1": pred_foreground,
            "pred2": pred_boundaries
        }
        util.visualize_data_napari(vis_data)


if __name__ == "__main__":
    visualize()
