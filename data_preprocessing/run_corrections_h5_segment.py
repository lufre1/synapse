#!/usr/bin/env python3
"""
Interactive correction / segmentation of mitochondria HDF5 files.

* Loads raw data, existing segmentation and any extra datasets you list in
  ``EXTRA_KEYS``.
* Shows them in a Napari viewer.
* Provides a dock widget with sliders / line‑edits that let you change every
  parameter of ``_segment_mitos``.
* “Run Segmentation” button calls the routine and adds the results as
  label / image layers.
* Standard save / paint / stop widgets are also available.
"""

import argparse
import os
from glob import glob

import h5py
import napari
import numpy as np
from magicgui import magicgui
from magicgui import magic_factory
from napari.utils.notifications import show_info
from tqdm import tqdm
from skimage.morphology import remove_small_holes, binary_erosion, remove_small_objects, binary_dilation
from skimage.measure import label

# ----------------------------------------------------------------------
#  Your own utilities (keep the imports you need)
# ----------------------------------------------------------------------
import elf.parallel as parallel
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
import synapse.util as util

# ----------------------------------------------------------------------
#  Paths – change them to whatever you need
# ----------------------------------------------------------------------
SAVE_DIR = "/home/freckmann15/data/mitochondria/mitopaper/volume-em-tiled-segs-corrected"
BASE_PATH = (
    "/home/freckmann15/data/mitochondria/mitopaper/volume-em-tiled-segs"
    "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/prel_model_on_4007"
)

# Datasets that you want to visualise in addition to the mandatory ones.
# They will be added as image layers and **also** used as the
# ``foreground`` and ``boundary`` inputs for the segmentation routine.
EXTRA_KEYS = ["pred/foreground", "pred/boundary", "labels/mitochondria"]


def _refresh_layer_choices(v: napari.Viewer) -> None:
    """
    Populate the ComboBox choices of all annotation widgets
    (Fill Holes, Erode Object, Dilate Object, Remove Small Objects)
    with the names of every label layer currently present in the viewer.

    If a layer named ``"segmentation"`` exists, make it the default
    selection; otherwise keep the current selection if it is still valid,
    or fall back to the first available label layer.
    """
    # ---- collect label‑layer names ------------------------------------
    label_names = [
        layer.name
        for layer in v.layers
        if isinstance(layer, napari.layers.Labels)
    ]

    # ---- helper to update a single widget ----------------------------
    def _update_widget(widget):
        # set the list of choices first
        widget.layer_name.choices = label_names

        # decide which value to show
        current = widget.layer_name.value
        if current in label_names:                     # still valid → keep it
            return

        # try to select “segmentation” if it exists
        if "segmentation" in label_names:
            widget.layer_name.value = "segmentation"
        # otherwise pick the first label layer (if any)
        elif label_names:
            widget.layer_name.value = label_names[0]
        else:
            widget.layer_name.value = None               # no layers → empty

    # ---- apply to every widget ---------------------------------------
    _update_widget(fill_holes)
    _update_widget(erode_object)
    _update_widget(dilate_object)
    _update_widget(remove_small_objects_widget)


@magicgui(
    call_button="Dilate Object",
    layer_name={"label": "Target label layer", "widget_type": "ComboBox"},
    object_id={"label": "Object ID", "widget_type": "LineEdit"},
    iterations={"label": "Dilation iterations", "widget_type": "LineEdit"},
)
def dilate_object(
    viewer: napari.Viewer,
    layer_name: str,
    object_id: str,
    iterations: str,
) -> None:
    """Dilate a single 3‑D object by the given number of iterations."""
    # ---- validation -------------------------------------------------
    if layer_name not in viewer.layers:
        return
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.Labels):
        return

    try:
        obj_id = int(object_id)
        iters = int(iterations)
    except ValueError:
        return

    # ---- mask of the selected object --------------------------------
    mask = layer.data == obj_id

    # ---- perform dilation -------------------------------------------
    dilated = mask
    for _ in range(iters):
        dilated = binary_dilation(dilated)
        if not dilated.any():          # object vanished – stop early
            break

    # ---- write back -------------------------------------------------
    new_data = layer.data.copy()
    new_data[layer.data == obj_id] = 0   # remove original voxels
    new_data[dilated] = obj_id          # insert dilated voxels
    layer.data = new_data


@magicgui(
    call_button="Fill Holes",
    layer_name={"label": "Target label layer", "widget_type": "ComboBox"},
    object_id={"label": "Object ID", "widget_type": "LineEdit"},
    max_hole_size={"label": "Max hole size (voxels)", "widget_type": "LineEdit"},
    connectivity={"label": "Connectivity (1, 2, or 3)", "widget_type": "LineEdit"},
)
def fill_holes(
    viewer: napari.Viewer,
    layer_name: str,
    object_id: str,
    max_hole_size: str,
    connectivity: str,
) -> None:
    """Fill holes of a single label object slice‑by‑slice (2‑D).

    Parameters
    ----------
    viewer : napari.Viewer
        Active napari viewer.
    layer_name : str
        Name of the label layer to edit.
    object_id : str
        Integer ID of the object whose holes are to be filled.
    max_hole_size : str
        Maximum hole volume (in voxels) to fill.
    connectivity : str
        1, 2 or 3 – 2‑D pixel connectivity used for each slice.
    """
    # ------------------------------------------------------------------
    #  Validate inputs
    # ------------------------------------------------------------------
    if layer_name not in viewer.layers:
        return
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.Labels):
        return

    try:
        obj_id = int(object_id)
        hole_limit = int(max_hole_size)
        conn = int(connectivity)
        if conn not in (1, 2, 3):
            raise ValueError
    except ValueError:
        return

    # ------------------------------------------------------------------
    #  3‑D mask of the selected object
    # ------------------------------------------------------------------
    mask = layer.data == obj_id

    # ------------------------------------------------------------------
    #  Iterate over the Z dimension, apply 2‑D hole filling on each slice
    # ------------------------------------------------------------------
    filled = np.zeros_like(mask, dtype=bool)

    # mask.shape is (Z, Y, X) for a typical volumetric label image
    for z in range(mask.shape[0]):
        slice_mask = mask[z]                     # 2‑D boolean array (Y, X)
        # remove_small_holes works on 2‑D arrays; connectivity argument is
        # interpreted as pixel connectivity for that slice.
        filled_slice = remove_small_holes(
            slice_mask,
            area_threshold=hole_limit,
            connectivity=conn,
        )
        filled[z] = filled_slice

    # ------------------------------------------------------------------
    #  Write the result back to the label layer
    # ------------------------------------------------------------------
    new_data = layer.data.copy()
    new_data[filled] = obj_id
    layer.data = new_data


@magicgui(
    call_button="Erode Object",
    layer_name={"label": "Target label layer", "widget_type": "ComboBox"},
    object_id={"label": "Object ID", "widget_type": "LineEdit"},
    iterations={"label": "Erosion iterations", "widget_type": "LineEdit"},
)
def erode_object(viewer: napari.Viewer, layer_name: str, object_id: str, iterations: str) -> None:
    if layer_name not in viewer.layers:
        return
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.Labels):
        return
    try:
        obj_id = int(object_id)
        iters = int(iterations)
    except ValueError:
        return

    mask = layer.data == obj_id
    eroded = mask
    for _ in range(iters):
        eroded = binary_erosion(eroded)
        if not eroded.any():
            break

    new_data = layer.data.copy()
    new_data[layer.data == obj_id] = 0
    new_data[eroded] = obj_id
    layer.data = new_data


@magicgui(
    call_button="Remove Small Objects",
    layer_name={"label": "Target label layer", "widget_type": "ComboBox"},
    max_size={"label": "Maximum object size (voxels)", "widget_type": "LineEdit"},
)
def remove_small_objects_widget(
    viewer: napari.Viewer,
    layer_name: str,
    max_size: str,
) -> None:
    """Delete every labeled object whose voxel count ≤ *max_size*.

    The operation works on the whole 3‑D label image; connectivity is set to 3
    (full 3‑D neighbourhood) so that only spatially coherent voxels are
    considered a single object.
    """
    # ---- validate layer -------------------------------------------------
    if layer_name not in viewer.layers:
        return
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.Labels):
        return

    # ---- parse size -----------------------------------------------------
    try:
        size_limit = int(max_size)
        if size_limit < 0:
            raise ValueError
    except ValueError:
        return

    # ---- perform removal ------------------------------------------------
    # remove_small_objects works directly on a labeled array.
    # Objects smaller than *size_limit* are set to 0 (background).
    cleaned = remove_small_objects(
        layer.data,
        min_size=size_limit,
        connectivity=3,   # full 3‑D connectivity
    )

    # ---- write back -----------------------------------------------------
    layer.data = cleaned


# ----------------------------------------------------------------------
#  Register the new widgets in the viewer (inside run_correction)
# ----------------------------------------------------------------------
def _add_annotation_widgets(v: napari.Viewer) -> None:
    """Create the two widgets, add them to the UI and keep their layer list current."""
    # initial population (run after the mandatory layers have been added)
    _refresh_layer_choices(v)

    # keep the list up‑to‑date when layers are added / removed / renamed
    v.layers.events.inserted.connect(lambda _: _refresh_layer_choices(v))
    v.layers.events.removed.connect(lambda _: _refresh_layer_choices(v))
    v.layers.events.changed.connect(lambda _: _refresh_layer_choices(v))

    # add the widgets to the dock
    v.window.add_dock_widget(fill_holes)
    v.window.add_dock_widget(erode_object)
    v.window.add_dock_widget(dilate_object)
    v.window.add_dock_widget(remove_small_objects_widget)


# ----------------------------------------------------------------------
#  Helper functions
# ----------------------------------------------------------------------
def _read_h5(path: str, key: str, scale_factor: int = 1):
    """Read a dataset from an HDF5 file and optionally down‑sample in XY."""
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            img = f[key][:, ::scale_factor, ::scale_factor]
            print(f"{key} shape after down‑sampling", img.shape)
            return img
        except KeyError:
            print(f"Error: dataset '{key}' not found in {path}")
            return None


def _parse_triplet(txt: str) -> tuple[int, int, int]:
    """Convert a string like ``'32,256,256'`` → ``(32, 256, 256)``."""
    parts = [int(p.strip()) for p in txt.split(",")]
    if len(parts) != 3:
        raise ValueError("Exactly three comma‑separated integers required.")
    return tuple(parts)  # type: ignore[return-value]


# ----------------------------------------------------------------------
#  UI: widget that holds all the segmentation parameters
# ----------------------------------------------------------------------
@magicgui(
    # ----- mandatory data layers (just labels, no widget) -----
    foreground={"label": "Foreground layer", "widget_type": "Label"},
    boundary={"label": "Boundary layer",   "widget_type": "Label"},
    # ----- parameters that the user can change -------------
    block_shape={
        "label": "Block shape (z,y,x)",
        "widget_type": "LineEdit",
        "tooltip": "comma‑separated ints, e.g. 32,256,256",
        "value": "32,256,256",
    },
    halo={
        "label": "Halo (z,y,x)",
        "widget_type": "LineEdit",
        "tooltip": "comma‑separated ints, e.g. 16,48,48",
        "value": "16,48,48",
    },
    seed_distance={
        "label": "Seed distance (voxels)",
        "widget_type": "Slider",
        "min": 1,
        "max": 20,
        "step": 1,
        "value": 4,
    },
    boundary_threshold={
        "label": "Boundary threshold",
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 0.9,
        "step": 0.01,
        "value": 0.15,
    },
    foreground_threshold={
        "label": "Foreground threshold",
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 0.9,
        "step": 0.01,
        "value": 0.75,
    },
    min_size={
        "label": "Min object size",
        "widget_type": "Slider",
        "min": 0,
        "max": 20000,
        "step": 100,
        "value": 2000,
    },
    area_threshold={
        "label": "Area threshold (2‑D)",
        "widget_type": "Slider",
        "min": 0,
        "max": 5000,
        "step": 100,
        "value": 200,
    },
    post_iter={
        "label": "Postprocess-Iterations (2D)",
        "widget_type": "Slider",
        "min": 0,
        "max": 12,
        "step": 1,
        "value": 4,
    },
    post_iter3d={
        "label": "Postprocess-Iterations (3D)",
        "widget_type": "Slider",
        "min": 0,
        "max": 24,
        "step": 1,
        "value": 8,
    },
    layout="vertical",
    call_button=False,          # we will add a separate “Run Segmentation” button
)
def seg_params_widget(
    viewer: napari.Viewer,
    foreground: str,
    boundary: str,
    block_shape: str,
    halo: str,
    seed_distance: int,
    boundary_threshold: float,
    foreground_threshold: float,
    min_size: int,
    area_threshold: int,
    post_iter: int,
    post_iter3d: int
) -> None:
    """
    This function **does not run** the segmentation – it only exists so that
    magicgui can create the widgets and store the current values as attributes.
    The body can stay empty.
    """
    pass


# ----------------------------------------------------------------------
#  The actual segmentation routine (unchanged)
# ----------------------------------------------------------------------
# def _segment_mitos(
#     foreground: np.ndarray,
#     boundary: np.ndarray,
#     block_shape=(128, 256, 256),
#     halo=(32, 48, 48),
#     seed_distance=6,
#     boundary_threshold=0.25,
#     foreground_threshold=0.75,
#     min_size=2000,
#     area_threshold=200,
#     dist=None,
# ):
#     """Return a dict with segmentation, seeds, distance map and h‑map."""
#     # ------------------------------------------------------------------
#     #  The code you already had – no changes required
#     # ------------------------------------------------------------------
#     boundaries = boundary
#     if dist is None:
#         dist = parallel.distance_transform(
#             boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape
#         )
#     hmap = (dist.max() - dist) / (dist.max() + 1e-6)
#     # hmap[
#     #     np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)
#     # ] = (hmap + boundaries).max()
#     barrier_mask = np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)
#     hmap[barrier_mask] = 1.0

#     seeds = np.logical_and(foreground > foreground_threshold, dist > seed_distance)
#     # seeds = parallel.label(seeds, block_shape=block_shape, verbose=True, connectivity=1)
#     seeds = label(seeds, connectivity=2)
#     seeds = apply_size_filter(seeds, 250, verbose=True, block_shape=block_shape)
    

#     # mask = (foreground + boundaries) > 0.5
#     mask = (foreground + np.where(boundaries < boundary_threshold, boundaries, 0)) > 0.5  # take overlap
#     # mask = foreground > foreground_threshold
#     # mask = np.logical_or((foreground > foreground_threshold), (boundary > boundary_threshold))  # (boundaries > (1-boundary_threshold)))

#     seg = np.zeros_like(seeds)
#     seg = parallel.seeded_watershed(
#         hmap, seeds, block_shape=block_shape, out=seg, verbose=True, halo=halo, mask=mask
#     )
#     seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
#     seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)

#     return {
#         "segmentation": seg.astype(np.uint8),
#         "seeds": seeds.astype(np.uint8),
#         "dist": dist.astype(np.float32),
#         "hmap": hmap.astype(np.float32),
#         "mask": mask.astype(np.uint8),
#     }


# ----------------------------------------------------------------------
#  UI for a single file – correction + segmentation
# ----------------------------------------------------------------------
def run_correction(input_path: str, output_path: str, fname: str, downsample: int = 1) -> bool:
    """Open a Napari viewer for one HDF5 file and let the user correct / segment."""
    continue_correction = True

    # ------------------------------------------------------------------
    # Load the mandatory datasets
    # ------------------------------------------------------------------
    raw = _read_h5(input_path, "raw", scale_factor=downsample)
    mitos = _read_h5(input_path, "seg", scale_factor=downsample)
    cristae = _read_h5(input_path, "labels/cristae", scale_factor=downsample)

    # ------------------------------------------------------------------
    # Load the extra datasets (foreground & boundary) and keep a dict
    # ------------------------------------------------------------------
    extra_layers = {}
    for key in EXTRA_KEYS:
        data = _read_h5(input_path, key, scale_factor=downsample)
        if data is not None:
            extra_layers[key] = data
        else:
            print(f"Warning: extra key '{key}' not found – skipping.")

    # ------------------------------------------------------------------
    # Build the Napari viewer
    # ------------------------------------------------------------------
    v = napari.Viewer()
    v.add_image(raw, name="raw")
    v.add_labels(mitos, name="original seg")
    if cristae is not None:
        v.add_labels(cristae, name="cristae")

    # Add the extra layers **as image layers** and give them the exact names
    # that the segmentation UI expects.
    for key, data in extra_layers.items():
        if "pred" in key:
            layer_name = key.split("/")[-1]          # e.g. "foreground" or "boundary"
            v.add_image(data, name=layer_name, blending="additive")
        elif "mitochondria" in key:
            v.add_labels(data, name=key)

    v.title = f"Tomo: {fname}, mitochondria"

    # ------------------------------------------------------------------
    #  ----  UI widgets  ------------------------------------------------
    # ------------------------------------------------------------------
    @magicgui(call_button="Save segmented Mitochondria")
    def save_correction_mitos(v: napari.Viewer):
        """Write the *segmentation* layer (and raw) back to HDF5."""
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        seg = v.layers["segmentation"].data
        raw = v.layers["raw"].data
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "labels/mitochondria", shape=seg.shape, dtype=seg.dtype, compression="gzip"
            )
            ds[:] = seg
            ds_raw = f.require_dataset(
                "raw", shape=raw.shape, dtype=raw.dtype, compression="gzip"
            )
            ds_raw[:] = raw
        show_info(f"Saved mitochondria labels to {output_path}.")

    @magicgui(call_button="Paint New Instance [f]")
    def paint_new_mitos(v: napari.Viewer):
        """Switch the mitochondria layer to paint mode."""
        layer_name = "segmentation" if "segmentation" in v.layers else "mitos"
        layer = v.layers[layer_name]
        show_info(f"Switched to paint mode on layer '{layer_name}'.")
        max_label = np.max(layer.data)
        layer.selected_label = max_label + 1 if max_label != 0 else 1
        layer.mode = "paint"

    @magicgui(call_button="Save Mitochondria")
    def save_correction(v: napari.Viewer):
        """Write the *mitos* layer (and raw) back to HDF5."""
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
        """Write the cristae layer back to HDF5."""
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
        """Terminate the correction loop."""
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    # ------------------------------------------------------------------
    #  Run‑segmentation button
    # ------------------------------------------------------------------
    @magicgui(call_button="Run Segmentation")
    def run_segmentation(v: napari.Viewer):
        """Read the UI parameters, call ``_segment_mitos`` and add the results."""
        # ------------------------------------------------------------------
        # 1️⃣  Grab the foreground / boundary layers (they must exist)
        # ------------------------------------------------------------------
        if "foreground" not in v.layers or "boundary" not in v.layers:
            show_info("Both 'foreground' and 'boundary' layers must be present.")
            return

        foreground = v.layers["foreground"].data
        boundary = v.layers["boundary"].data

        # ------------------------------------------------------------------
        # 2️⃣  Pull the current values from the *parameter* widget
        # ------------------------------------------------------------------
        params = param_gui   # the FunctionGui instance created below

        try:
            block_shape = _parse_triplet(params.block_shape.value)
            halo = _parse_triplet(params.halo.value)
        except ValueError as e:
            show_info(str(e))
            return

        seed_distance = params.seed_distance.value
        boundary_threshold = params.boundary_threshold.value
        foreground_threshold = params.foreground_threshold.value
        min_size = params.min_size.value
        area_threshold = params.area_threshold.value
        post_iter = params.post_iter.value
        post_iter3d = params.post_iter3d.value

        # ------------------------------------------------------------------
        # 3️⃣  Call the real segmentation routine
        # ------------------------------------------------------------------
        seg_dict = util.segment_mitos(
            foreground=foreground,
            boundary=boundary,
            block_shape=block_shape,
            halo=halo,
            seed_distance=seed_distance,
            boundary_threshold=boundary_threshold,
            foreground_threshold=foreground_threshold,
            min_size=min_size,
            area_threshold=area_threshold,
            post_iter3d=0
            # dist=v.layers["dist"].data if "dist" in v.layers else None,
        )

        # ------------------------------------------------------------------
        # 4️⃣  Insert / update the returned layers
        # ------------------------------------------------------------------
        for name, data in seg_dict.items():
            if name in v.layers:
                v.layers[name].data = data
            else:
                # Float → image layer, everything else → label layer
                if np.issubdtype(data.dtype, np.floating):
                    v.add_image(data, name=name, blending="additive")
                else:
                    v.add_labels(data, name=name)
        _refresh_layer_choices(v)
        show_info("Segmentation finished – layers added/updated.")

    # ------------------------------------------------------------------
    #  Add everything to the Napari UI
    # ------------------------------------------------------------------
    # 1️⃣  Parameter widget – instantiate it **once** and keep a reference
    param_gui = seg_params_widget
    v.window.add_dock_widget(param_gui.native, area="right", name="Segmentation parameters")

    # 2️⃣  The “Run Segmentation” button
    v.window.add_dock_widget(run_segmentation)
    _add_annotation_widgets(v)

    # 3️⃣  The other utility widgets you already had
    v.window.add_dock_widget(paint_new_mitos)
    # v.window.add_dock_widget(save_correction)
    # v.window.add_dock_widget(save_correction_cristae)
    v.window.add_dock_widget(save_correction_mitos)
    v.window.add_dock_widget(stop_correction)

    # ------------------------------------------------------------------
    #  Keyboard shortcuts (optional)
    # ------------------------------------------------------------------
    v.bind_key("s", lambda _: save_correction(v))
    v.bind_key("q", lambda _: stop_correction(v))
    v.bind_key("f", lambda _: paint_new_mitos(v))
    v.bind_key("r", lambda _: run_segmentation(v))
    _refresh_layer_choices(v)
    napari.run()
    return continue_correction


# ----------------------------------------------------------------------
#  Loop over all files in the dataset
# ----------------------------------------------------------------------
def correct_mitochondria(args):
    base_path = args.base_path
    if args.save_dir is None:
        save_dir = base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True)) if os.path.isdir(base_path) else [base_path]

    iteration = 0
    continue_from = "block_z000512_000640_y001936_003872_x003312_004968.h5"
    continue_now = True

    for path in tqdm(file_paths):
        # if "refined" in os.path.basename(path):
        #     print("Skip because it is already refined")
        #     continue
        if continue_from in path:
            continue_now = True
        if not continue_now:
            continue

        iteration += 1
        _, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        fname = fname.replace("_raw", "") + "_refined.h5"
        output_path = os.path.join(save_dir, fname)

        if os.path.exists(output_path) and not args.force_overwrite:
            print(f"Already exists:\n{output_path}\n")
            continue

        print(f"Loading:\n{path}\nwill save to:\n{output_path}\n")

        if not run_correction(path, output_path, fname, args.downsample):
            break


# ----------------------------------------------------------------------
#  CLI entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        "-b",
        type=str,
        default=BASE_PATH,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        default=None,
        help="Path to save the corrected data",
    )
    parser.add_argument(
        "--force_overwrite",
        "-f",
        action="store_true",
        help="Overwrite existing segmentation results",
    )
    parser.add_argument("--downsample", "-ds", type=int, default=1, help="Downsampling factor for the data.")
    args = parser.parse_args()
    correct_mitochondria(args)


if __name__ == "__main__":
    main()