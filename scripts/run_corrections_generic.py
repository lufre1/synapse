import h5py
import mrcfile
import os
from glob import glob
import numpy as np
import napari
import argparse
from magicgui import magicgui
from skimage.measure import label
# from scipy import ndimage
from napari.utils.notifications import show_info
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes, binary_erosion, binary_dilation
# from skimage.measure import label, regionprops
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
import elf.parallel as parallel
import synapse.h5_util as hutil


def _refresh_layer_choices(v: napari.Viewer, widget=None) -> None:
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
    def _update_widget(w):
        try:
            w.layer_name.choices = label_names
        except (RuntimeError, AttributeError):
            # Widget might be deleted; skip updating
            return

        # decide which value to show
        current = w.layer_name.value
        if current in label_names:                     # still valid → keep it
            return

        # try to select “segmentation” if it exists
        if "segmentation" in label_names:
            w.layer_name.value = "segmentation"
        # otherwise pick the first label layer (if any)
        elif label_names:
            w.layer_name.value = label_names[0]
        else:
            w.layer_name.value = None               # no layers → empty

    try:
        if widget is None:
            _update_widget(fill_holes)
            _update_widget(erode_object)
            _update_widget(dilate_object)
            _update_widget(remove_small_objects_widget_impl)
        else:
            _update_widget(widget)
    except Exception:
        pass


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
    show_info(f"Filled holes in object {obj_id} in layer {layer_name}.")


@magicgui(
    call_button="Remove Small Objects",
    layer_name={"label": "Target label layer", "widget_type": "ComboBox"},
    max_size={"label": "Maximum object size (voxels)", "widget_type": "LineEdit"},
)
def remove_small_objects_widget_impl(
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
    show_info(f"Removed objects smaller than {size_limit} voxels in layer {layer_name}.")


def _add_annotation_widgets(v: napari.Viewer, widget_instance) -> None:
    """Create the two widgets, add them to the UI and keep their layer list current."""
    # initial population (run after the mandatory layers have been added)
    _refresh_layer_choices(v, widget_instance)

    # keep the list up‑to‑date when layers are added / removed / renamed
    v.layers.events.inserted.connect(lambda _: _refresh_layer_choices(v, widget_instance))
    v.layers.events.removed.connect(lambda _: _refresh_layer_choices(v, widget_instance))
    v.layers.events.changed.connect(lambda _: _refresh_layer_choices(v, widget_instance))
    v.window.add_dock_widget(widget_instance)
    _refresh_layer_choices(v, widget_instance)


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


def run_correction(input_path, output_path, fname, raw_key="raw", scale=1):
    """Open a Napari viewer for a single HDF5 file and allow manual correction."""
    continue_correction = True

    # ------------------------------------------------------------------
    # Load the mandatory datasets
    # ------------------------------------------------------------------
    data = {}
    data = hutil.read_data(input_path, scale=scale)
    vs = hutil.read_voxel_size(input_path) 
    if scale != 1:
        vs = {k: v * scale for k, v in vs.items()}

    v = napari.Viewer()

    for key, val in data.items():
        if "raw" in key or "pred" in key:
            v.add_image(val, name="raw")
        else:
            v.add_labels(val, name=key)
    raw_layer = next((layer for layer in v.layers if "raw" in layer.name), None)
    if raw_layer:
        # Remove the "raw" layer from its current position
        v.layers.remove(raw_layer)
        # Add the "raw" layer to the beginning of the layer list
        v.layers.insert(0, raw_layer)
    v.title = f"{fname}"

    # ------------------------------------------------------------------
    # UI widgets -------------------------------------------------------
    # ------------------------------------------------------------------
    @magicgui(call_button="Save All")
    def save_all(v: napari.Viewer):
        """Write the new mitochondria (and raw) data back to the HDF5 file."""
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        for layer in v.layers:
            with h5py.File(output_path, "a") as f:
                data = layer.data
                ds = f.require_dataset(
                    layer.name,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression="gzip",
                )
                ds[:] = data
                if "raw" in layer.name and vs is not None:
                    voxel_size_array = vs if isinstance(vs, np.ndarray) else np.array(vs, dtype=np.float32)
                    ds.attrs.create(name='voxel_size', data=voxel_size_array)
        show_info(f"Saved all datasets to {output_path}.")

    @magicgui(
        call_button="Paint New Instance [f]",
        layer_name={"label": "Target layer", "widget_type": "ComboBox"},
    )
    def paint_new_instance(v: napari.Viewer, layer_name: str):
        """Switch the selected layer to paint mode and assign a new unique ID."""
        if layer_name not in v.layers:
            show_info(f"Could not find layer '{layer_name}'.")
            return
        
        layer = v.layers[layer_name]
        
        # Get all unique values in the layer data
        uniqs = np.unique(layer.data)
        uniqs = uniqs[uniqs != 0]  # Exclude background (0)
        
        # Find the maximum ID and add 1 for the new ID
        if len(uniqs) > 0:
            new_id = np.max(uniqs) + 1
        else:
            new_id = 1  # Start with 1 if no labels exist
        
        show_info(f"Switched to paint mode on layer '{layer_name}' with new ID {new_id}.")
        layer.selected_label = new_id
        layer.mode = "paint"

    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        """Terminate the correction loop."""
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    @magicgui(
        call_button="Relabel instances",
        layer_name={"label": "Target layer", "widget_type": "ComboBox"},
    )
    def relabel(v: napari.Viewer, layer_name: str):
        if layer_name not in v.layers:
            show_info(f"Could not find layer '{layer_name}'.")
            return
        layer = v.layers[layer_name]
        layer.data = label(layer.data)

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

    # Create a fresh instance of the remove small objects widget for this viewer
    remove_small_objects_widget = remove_small_objects_widget_impl.copy()
    remove_holes_widget = fill_holes.copy()
    dilate_object_widget = dilate_object.copy()
    erode_object_widget = erode_object.copy()
    relabel_widget = relabel.copy()
    
    # Add all widgets to the UI
    v.window.add_dock_widget(paint_new_instance)
    _refresh_layer_choices(v, paint_new_instance)
    _add_annotation_widgets(v, dilate_object_widget)
    _add_annotation_widgets(v, erode_object_widget)
    _add_annotation_widgets(v, remove_small_objects_widget)
    _add_annotation_widgets(v, remove_holes_widget)
    _add_annotation_widgets(v, relabel_widget)

    v.window.add_dock_widget(save_all)
    v.window.add_dock_widget(stop_correction)
    # v.window.add_dock_widget(run_segmentation)   # <<< NEW >>>

    v.bind_key("s", lambda _: save_all(v))
    v.bind_key("q", lambda _: stop_correction(v))
    v.bind_key("f", lambda _: paint_new_instance(v))
    # v.bind_key("r", lambda _: run_segmentation(v))   # optional shortcut for segmentation

    napari.run()
    # _refresh_layer_choices(v)

    return continue_correction


def correct_mitochondria(args):
    base_path = args.base_path
    save_dir = args.save_dir if args.save_dir is not None else base_path
    os.makedirs(save_dir, exist_ok=True)

    # raw_files = sorted(glob(os.path.join(base_path, "**/*raw.mrc"), recursive=True))
    if os.path.isdir(base_path):
        file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    else:
        file_paths = [base_path]
    # file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))

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
    parser.add_argument("--base_path", "-b", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_dir", "-s", type=str, default=None, help="Path to save the data to")
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Whether to over-write already present segmentation results.")
    parser.add_argument("--scale", "-sc", type=int, default=1, help="Downsampling factor for the data.")
    args = parser.parse_args()

    correct_mitochondria(args)


if __name__ == "__main__":
    main()