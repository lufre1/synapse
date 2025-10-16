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
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
import elf.parallel as parallel

# SAVE_DIR = "/home/freckmann15/data/mitochondria/corrected_mitos"
# BASE_PATH = "/home/freckmann15/data/mitochondria/corrected_mito_h5"
SAVE_DIR = "/home/freckmann15/data/mitochondria/mitopaper/volume-em-tiled-segs-corrected"
BASE_PATH = "/home/freckmann15/data/mitochondria/mitopaper/volume-em-tiled-segs/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/prel_model_on_4007"
EXTRA_KEYS = ["pred/foreground", "pred/boundary"]


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


def _segment_mitos(foreground: np.ndarray,
                   boundary: np.ndarray,
                   block_shape=(32, 256, 256),
                   halo=(16, 48, 48),
                   seed_distance=6,
                   boundary_threshold=0.25-0.1, # - 0.1,
                   min_size=2000,
                   area_threshold=200 * 1,
                   dist=None
                   ):
    foreground, boundaries = foreground, boundary
    # # #boundaries = binary_erosion(boundaries < boundary_threshold, structure=np.ones((1, 3, 3)))
    if dist is None:
        dist = parallel.distance_transform(boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    # # data["pred_dist_without_fore"] = parallel.distance_transform((boundaries) < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    hmap = ((dist.max() - dist) / dist.max())

    hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()

    # # hmap = hmap.clip(min=0)
    seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=True)
    # # #seeds = binary_fill_holes(seeds)

    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=True, halo=halo,
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    return {
        "segmentation": seg.astype(np.uint8),
        "seeds": seeds.astype(np.uint8),
        "dist": dist.astype(np.float32),
        "hmap": hmap.astype(np.float32)
    }


# def _segment_mitos(foreground: np.ndarray,
#                    boundary: np.ndarray,
#                    block_shape=(128, 128, 128),
#                    threshold=0.5,
#                    min_size=5000,
#                    area_threshold=500,):
#     foreground_mask = np.where(foreground > threshold, 1, 0)
#     boundary_mask = np.where(boundary > threshold-0.25, 1, 0)
#     mask = np.logical_or(foreground_mask, np.logical_and(foreground_mask, boundary_mask))
#     seg = label(mask) #, block_shape=block_shape)
#     seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
#     seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
#     return seg
# ----------------------------------------------------------------------


def run_correction(input_path, output_path, fname):
    """Open a Napari viewer for a single HDF5 file and allow manual correction."""
    continue_correction = True

    # ------------------------------------------------------------------
    # Load the mandatory datasets
    # ------------------------------------------------------------------
    raw = _read_h5(input_path, "data")
    mitos = _read_h5(input_path, "seg")
    # mitos.setflags(write=1)

    cristae = _read_h5(input_path, "labels/cristae")
    if cristae is not None:
        cristae.setflags(write=1)

    # ------------------------------------------------------------------
    # <<< NEW >>> Load any extra datasets the user asked for
    # ------------------------------------------------------------------
    extra_layers = {}
    for key in EXTRA_KEYS:
        data = _read_h5(input_path, key)
        if data is not None:
            data.setflags(write=1)
            extra_layers[key] = data
        else:
            print(f"Warning: extra key '{key}' not found – skipping.")

    # ------------------------------------------------------------------
    # Build the Napari UI
    # ------------------------------------------------------------------
    v = napari.Viewer()

    v.add_image(raw, name="raw")
    v.add_labels(mitos, name="mitos")
    if cristae is not None:
        v.add_labels(cristae, name="cristae")

    # Add the extra layers (they will appear as image layers)
    for key, data in extra_layers.items():
        # Use only the last component of the HDF5 path as the layer name
        layer_name = key.split("/")[-1]
        v.add_image(data, name=layer_name, blending="additive")

    v.title = f"Tomo: {fname}, mitochondria"

    # ------------------------------------------------------------------
    # UI widgets --------------------------------------------------------
    # ------------------------------------------------------------------
    @magicgui(call_button="Save segmented Mitochondria")
    def save_correction_mitos(v: napari.Viewer):
        """Write the new mitochondria (and raw) data back to the HDF5 file."""
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        mitos = v.layers["segmentation"].data
        raw = v.layers["raw"].data
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "labels/mitochondria",
                shape=mitos.shape,
                dtype=mitos.dtype,
                compression="gzip",
            )
            ds[:] = mitos
            ds_raw = f.require_dataset(
                "raw", shape=raw.shape, dtype=raw.dtype, compression="gzip"
            )
            ds_raw[:] = raw
        show_info(f"Saved mitochondria labels to {output_path}.")

    @magicgui(call_button="Paint New Vesicle [f]")
    def paint_new_mitos(v: napari.Viewer):
        """Switch the mitochondria layer to paint mode."""
        layer = v.layers["mitos"]
        layer.selected_label = 1
        layer.mode = "paint"

    @magicgui(call_button="Save Mitochondria")
    def save_correction(v: napari.Viewer):
        """Write the current mitochondria (and raw) data back to the HDF5 file."""
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        mitos = v.layers["mitos"].data
        raw = v.layers["raw"].data
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "labels/mitochondria",
                shape=mitos.shape,
                dtype=mitos.dtype,
                compression="gzip",
            )
            ds[:] = mitos
            ds_raw = f.require_dataset(
                "raw", shape=raw.shape, dtype=raw.dtype, compression="gzip"
            )
            ds_raw[:] = raw
        show_info(f"Saved mitochondria labels to {output_path}.")

    @magicgui(call_button="Save Cristae")
    def save_correction_cristae(v: napari.Viewer):
        """Write the cristae layer back to the HDF5 file."""
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        cristae = v.layers["cristae"].data
        with h5py.File(output_path, "a") as f:
            ds_c = f.require_dataset(
                "labels/cristae",
                shape=cristae.shape,
                dtype=cristae.dtype,
                compression="gzip",
            )
            ds_c[:] = cristae
        show_info(f"Saved cristae labels to {output_path}.")

    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        """Terminate the correction loop."""
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    # <<< NEW >>> ---------------------------------------------------------
    @magicgui(call_button="Run Segmentation")
    def run_segmentation(v: napari.Viewer):
        """
        Run the (stub) segmentation routine on the raw data and add the
        result as a new label layer called ``segmentation``.
        """
        foreground = v.layers["foreground"].data
        boundary = v.layers["boundary"].data
        seg = _segment_mitos(foreground=foreground, boundary=boundary)
        # breakpoint()
        # If a segmentation layer already exists, replace its data.
        if seg is np.ndarray:  # check if the segmentation is a numpy array
            print("np array")
            if "segmentation" in v.layers:
                v.layers["segmentation"].data = seg
            else:
                v.add_labels(seg, name="segmentation")
        elif isinstance(seg, dict):  # check if the segmentation is a dictionary
            print("dict ")
            for k, val in seg.items():
                if k in v.layers:
                    v.layers[k].data = val
                elif np.issubdtype(val.dtype, np.floating):
                    v.add_image(val, name=k, blending="additive")
                else:
                    v.add_labels(val, name=k)
                    
        show_info("Segmentation finished – layer 'segmentation' added/updated.")
    # ------------------------------------------------------------------

    # Add all widgets to the UI
    v.window.add_dock_widget(paint_new_mitos)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(save_correction_cristae)
    v.window.add_dock_widget(save_correction_mitos)
    v.window.add_dock_widget(stop_correction)
    v.window.add_dock_widget(run_segmentation)   # <<< NEW >>>

    v.bind_key("s", lambda _: save_correction(v))
    v.bind_key("q", lambda _: stop_correction(v))
    v.bind_key("f", lambda _: paint_new_mitos(v))
    v.bind_key("r", lambda _: run_segmentation(v))   # optional shortcut for segmentation

    napari.run()

    return continue_correction


def correct_mitochondria(args):
    base_path = args.base_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # raw_files = sorted(glob(os.path.join(base_path, "**/*raw.mrc"), recursive=True))
    file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))

    iteration = 0
    continue_from = "36859_J1_66K_TS_PS_01_rec_2kb1dawbp_crop_downscaled.h5"
    continue_now = True
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
    parser.add_argument("--base_path", "-b", type=str, default=BASE_PATH, help="Path to the data directory")
    parser.add_argument("--save_dir", "-s", type=str, default=SAVE_DIR, help="Path to save the data to")
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Whether to over-write already present segmentation results.")
    args = parser.parse_args()

    correct_mitochondria(args)


if __name__ == "__main__":
    main()
