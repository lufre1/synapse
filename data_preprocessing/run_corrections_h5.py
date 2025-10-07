import h5py
import mrcfile
import os
from glob import glob
import numpy as np
import napari
import argparse
from magicgui import magicgui
from skimage.measure import label
from scipy import ndimage
from napari.utils.notifications import show_info
from tqdm import tqdm
from skimage.measure import label, regionprops

SAVE_DIR = "/home/freckmann15/data/mitochondria/corrected_mitos"
BASE_PATH = "/home/freckmann15/data/mitochondria/corrected_mito_h5"


def _read_h5(path, key, scale_factor=1):
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


def run_correction(input_path, output_path, fname):
    continue_correction = True
    raw = _read_h5(input_path, "raw")
    mitos = _read_h5(input_path, "labels/mitochondria")
    mitos.setflags(write=1)
    cristae = _read_h5(input_path, "labels/cristae")
    if cristae is not None:
        cristae.setflags(write=1)

    v = napari.Viewer()

    # if orig_labels is None:
    #     orig_labels = labels

    v.add_image(raw)
    v.add_labels(mitos)
    labels_layer = v.layers["mitos"]
    labels_layer.mode = "paint"
    if cristae is not None:
        v.add_labels(cristae)
    
    v.title = f"Tomo: {fname}, mitochondria"

    @magicgui(call_button="Paint New Vesicle [f]")
    def paint_new_mitos(v: napari.Viewer):
        layer = v.layers["labels"]
        layer.selected_label = 1
        layer.mode = "paint"

    @magicgui(call_button="Save Mitochondria")
    def save_correction(v: napari.Viewer):
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        mitos = v.layers["mitos"].data
        raw = v.layers["raw"].data
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "labels/mitochondria", shape=mitos.shape, dtype=mitos.dtype, compression="gzip"
            )
            ds[:] = mitos
            ds_raw = f.require_dataset(
                "raw", shape=raw.shape, dtype=raw.dtype, compression="gzip"
            )
            ds_raw[:] = raw
        show_info(f"Saved mitochondria labels to {output_path}.")

    @magicgui(call_button="Save Cristae")
    def save_correction_cristae(v: napari.Viewer):
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        cristae = v.layers["cristae"].data

        with h5py.File(output_path, "a") as f:
            ds_c = f.require_dataset(
                    "labels/cristae", shape=cristae.shape, dtype=cristae.dtype, compression="gzip"
                )
            ds_c[:] = cristae
            show_info(f"Saved cristae labels to {output_path}.")
    
    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    v.window.add_dock_widget(paint_new_mitos)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(save_correction_cristae)
    v.window.add_dock_widget(stop_correction)

    v.bind_key("s", lambda _:  save_correction(v))
    v.bind_key("q", lambda _:  stop_correction(v))
    v.bind_key("f", lambda _:  paint_new_mitos(v))

    napari.run()

    return continue_correction


def correct_mitochondria(args):
    base_path = args.base_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    #raw_files = sorted(glob(os.path.join(base_path, "**/*raw.mrc"), recursive=True))
    file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    
    iteration = 0
    continue_from = "M2_eb3_model.h5"
    continue_now = False
    for path in tqdm(file_paths):
        if continue_from in path:
            continue_now = True
        if not continue_now:
            continue
        iteration += 1
        old_path, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        fname = fname.replace("_raw", "")
        fname = fname + ".h5"
        #print("\nfname: ", fname)
        output_path = os.path.join(save_dir, fname)
        if iteration <= 0:
            continue

        if os.path.exists(output_path) and not args.force_overwrite:
            print(f"Already exists: \n{output_path}\n")
            continue
        print(f"Loading: \n{path} \nwould save to: \n{output_path}\n")

        if not run_correction(path, output_path, fname):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=BASE_PATH, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Path to save the data to")
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Whether to over-write already present segmentation results.")
    args = parser.parse_args()

    correct_mitochondria(args)


if __name__ == "__main__":
    main()