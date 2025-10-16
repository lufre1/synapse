import argparse
import os
from glob import glob

import h5py
import napari
import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_info
from tqdm import tqdm
import elf.parallel as parallel
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d

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
EXTRA_KEYS = ["pred/foreground", "pred/boundary"]


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
        "value": 6,
    },
    boundary_threshold={
        "label": "Boundary threshold",
        "widget_type": "FloatSlider",
        "min": 0.0,
        "max": 0.5,
        "step": 0.01,
        "value": 0.15,
    },
    min_size={
        "label": "Min object size",
        "widget_type": "Slider",
        "min": 0,
        "max": 10000,
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
    recompute={
        "label": "Re‑calculate everything",
        "widget_type": "CheckBox",
        "value": False,
        "tooltip": "Force recomputation of distance‑transform, h‑map and seeds",
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
    min_size: int,
    area_threshold: int,
    recompute: bool,
) -> None:
    """
    This function **does not run** the segmentation – it only exists so that
    magicgui can create the widgets and store the current values as attributes.
    The body can stay empty.
    """
    pass


import tempfile
import warnings
import numpy as np
import h5py


def _read_h5_memmap(
    path: str,
    key: str,
    dtype: np.dtype = np.float32,
    scale_factor: int = 1,
):
    """
    Return a **read‑only** ``np.memmap`` that points to the HDF5 dataset
    ``key`` inside ``path``.  If the dataset does not exist, ``None`` is
    returned (mirroring the behaviour of the original ``_read_h5``).

    The function works for both *contiguous* and *chunked* datasets
    (see the long docstring in the previous answer for details).
    """
    import warnings, tempfile

    # --------------------------------------------------------------
    # 1️⃣  Open the file – if the key is missing we return None.
    # --------------------------------------------------------------
    try:
        f = h5py.File(path, "r")
        dset = f[key]                     # may raise KeyError
    except KeyError:
        # The dataset simply isn’t there – behave like the old helper.
        # (We keep the file closed automatically because we never opened it.)
        return None

    # --------------------------------------------------------------
    # 2️⃣  Compute XY down‑sampling.
    # --------------------------------------------------------------
    shape = (
        dset.shape[0],
        dset.shape[1] // scale_factor,
        dset.shape[2] // scale_factor,
    )

    # --------------------------------------------------------------
    # 3️⃣  Try the fast path: a contiguous dataset gives us a byte offset.
    # --------------------------------------------------------------
    try:
        offset = dset.id.get_offset()
    except Exception:                     # some HDF5 drivers raise on get_offset()
        offset = None

    if offset is not None:
        # ----- Contiguous storage – true mem‑map -----
        mm = np.memmap(
            filename=f.filename,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="C",
        )
        if scale_factor != 1:
            mm = mm[:, ::scale_factor, ::scale_factor]
        # Keep the HDF5 file alive while the mem‑map exists.
        mm._h5_file = f                     # type: ignore[attr-defined]
        return mm

    # --------------------------------------------------------------
    # 4️⃣  Chunked storage – fall back to a temporary on‑disk copy.
    # --------------------------------------------------------------
    warnings.warn(
        f"Dataset '{key}' in '{path}' is chunked.  Creating a temporary "
        "on‑disk copy so that a mem‑map can be used.  This adds a small "
        "disk overhead but keeps RAM usage low.",
        RuntimeWarning,
    )

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_name = tmp.name
    tmp.close()

    # Allocateritable) with the correct shape.
    mm = np.memmap(
        filename=tmp_name,
        dtype=dtype,
        mode="w+",
        shape=shape,
        order="C",
    )

    # Copy the data chunk‑by‑chunk.
    if scale_factor == 1:
        dset.read_direct(mm)
    else:
        full_buf = np.empty(dset.shape, dtype=dtype)
        dset.read_direct(full_buf)
        mm[:] = full_buf[:, ::scale_factor, ::scale_factor]

    # Close the writable view and reopen as read‑only.
    del mm
    mm = np.memmap(
        filename=tmp_name,
        dtype=dtype,
        mode="r",
        shape=shape,
        order="C",
    )
    mm._tmp_file = tmp_name                # type: ignore[attr-defined]

    # The original HDF5 file can now be closed – we no longer need it.
    f.close()
    return mm


def _segment_mitos_mem(
    foreground: np.ndarray,
    boundary:   np.ndarray,
    out_path:   str,               # HDF5 file where the final seg will be stored
    block_shape=(32, 256, 256),
    halo=(16, 48, 48),
    seed_distance=6,
    boundary_threshold=0.15,
    min_size=2000,
    area_threshold=200,
    recompute: bool = True,
):
    """
    Memory‑efficient segmentation that **re‑uses** the distance transform,
    h‑map and seed map if they already exist in the file.

    The intermediates are kept in the HDF5 group ``/intermediate``.
    On the first call they are created; on later calls they are read back
    (no expensive recomputation).
    """
    import gc, os, warnings, tempfile
    import numpy as np, h5py, elf.parallel as parallel
    from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d

    # ------------------------------------------------------------------
    # 1️⃣  Open the file for writing the final segmentation (plain “a” mode)
    # ------------------------------------------------------------------
    with h5py.File(out_path, "a") as f_out:
        # ---- final segmentation dataset ---------------------------------
        seg_ds = f_out.require_dataset(
            "labels/mitochondria",
            shape=foreground.shape,
            dtype=np.uint8,
            compression="gzip",
        )

        # ----- get a mem‑map for the final label image ------------------
        try:
            seg_offset = seg_ds.id.get_offset()
        except Exception:
            seg_offset = None

        if seg_offset is not None:                     # contiguous → true mem‑map
            seg_mem = np.memmap(
                filename=f_out.filename,
                dtype=np.uint8,
                mode="r+",
                offset=seg_offset,
                shape=foreground.shape,
                order="C",
            )
            tmp_file_for_seg = None
        else:                                          # chunked → temporary file
            warnings.warn(
                "Output dataset is chunked – using a temporary file for the "
                "segmentation (still memory‑efficient).",
                RuntimeWarning,
            )
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_name = tmp.name
            tmp.close()
            seg_mem = np.memmap(
                filename=tmp_name,
                dtype=np.uint8,
                mode="w+",
                shape=foreground.shape,
                order="C",
            )
            tmp_file_for_seg = tmp_name

        # ------------------------------------------------------------------
        # 2️⃣  Helper that returns a **writable** mem‑map for a dataset.
        #     If the dataset already exists we return a **read‑only** view.
        # ------------------------------------------------------------------
        grp = f_out.require_group("intermediate")

                # ------------------------------------------------------------------
        # 2️⃣  Helper that returns a mem‑map for an intermediate.
        # ------------------------------------------------------------------
        def _get_memmap(name: str, dtype):
            """
            If ``recompute`` is True we ignore any existing dataset and
            create a **new writable** mem‑map (which will later be copied back
            into the HDF5 file).  Otherwise we reuse an existing read‑only view.
            """
            # ---- 1)  Force‑re‑create ? ---------------------------------
            if recompute and name in grp:
                # delete the old dataset – we will write a fresh one
                del grp[name]

            # ---- 2)  Does the dataset already exist ? -------------------
            if name in grp:                     # already on disk → read‑only view
                dset = grp[name]
                try:
                    off = dset.id.get_offset()
                except Exception:
                    off = None
                if off is not None:            # contiguous storage
                    return np.memmap(
                        filename=f_out.filename,
                        dtype=dtype,
                        mode="r",
                        offset=off,
                        shape=foreground.shape,
                        order="C",
                    )
                # chunked → copy to a temporary file and return a read‑only view
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp_name = tmp.name
                tmp.close()
                mm = np.memmap(
                    filename=tmp_name,
                    dtype=dtype,
                    mode="w+",
                    shape=foreground.shape,
                    order="C",
                )
                dset.read_direct(mm)            # stream the data
                del mm
                return np.memmap(
                    filename=tmp_name,
                    dtype=dtype,
                    mode="r",
                    shape=foreground.shape,
                    order="C",
                )
            # ---- 3)  No dataset → create a writable one -----------------
            else:
                dset = grp.create_dataset(
                    name, shape=foreground.shape, dtype=dtype, compression="gzip"
                )
                try:
                    off = dset.id.get_offset()
                except Exception:
                    off = None
                if off is not None:            # contiguous → direct writable mem‑map
                    return np.memmap(
                        filename=f_out.filename,
                        dtype=dtype,
                        mode="r+",
                        offset=off,
                        shape=foreground.shape,
                        order="C",
                    )
                # chunked → temporary file that will be copied back later
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp_name = tmp.name
                tmp.close()
                mm = np.memmap(
                    filename=tmp_name,
                    dtype=dtype,
                    mode="w+",
                    shape=foreground.shape,
                    order="C",
                )
                # keep a reference so the caller can copy it back
                mm._tmp_file = tmp_name
                mm._dset = dset
                return mm


        # --------------------------------------------------------------
        # 3️⃣  Distance transform (reuse if present)
        # --------------------------------------------------------------
        dist = _get_memmap("dist", np.float32)
        if hasattr(dist, "_tmp_file"):          # newly created → we must compute it
            dist_result = parallel.distance_transform(
                (boundary < boundary_threshold),
                halo=halo,
                verbose=True,
                block_shape=block_shape,
            )
            dist[:] = dist_result
            # copy back to the real HDF5 dataset (chunked case)
            dist._dset[:] = dist[:]
            os.remove(dist._tmp_file)

        # --------------------------------------------------------------
        # 4️⃣  h‑map (reuse if present)
        # --------------------------------------------------------------
        hmap = _get_memmap("hmap", np.float32)
        if hasattr(hmap, "_tmp_file"):          # need to compute
            max_dist = dist.max()
            hmap_arr = (max_dist - dist) / max_dist
            mask = np.logical_and(
                boundary > boundary_threshold, foreground < boundary_threshold
            )
            hmap_arr[mask] = (hmap_arr + boundary)[mask].max()
            hmap[:] = hmap_arr
            hmap._dset[:] = hmap[:]
            os.remove(hmap._tmp_file)

        # --------------------------------------------------------------
        # 5️⃣  Seeds (reuse if present)
        # --------------------------------------------------------------
        seeds = _get_memmap("seeds", np.uint8)
        if hasattr(seeds, "_tmp_file"):        # need to compute
            seeds_arr = np.logical_and(foreground > 0.5, dist > seed_distance)
            seeds_arr = parallel.label(
                seeds_arr, block_shape=block_shape, verbose=True
            )
            seeds[:] = seeds_arr
            seeds._dset[:] = seeds[:]
            os.remove(seeds._tmp_file)

        # --------------------------------------------------------------
        # 6️⃣  Watershed – correct argument order (out is the 3rd positional)
        # --------------------------------------------------------------
        mask = (foreground + boundary) > 0.5
        parallel.seeded_watershed(
            hmap,
            seeds,
            seg_mem,          # <-- output array (positional)
            block_shape,      # <-- block shape (positional)
            mask=mask,
            verbose=True,
            halo=halo,
        )

        # --------------------------------------------------------------
        # 7️⃣  Size filter – copy result back into seg_mem
        # --------------------------------------------------------------
        seg_mem[:] = apply_size_filter(
            seg_mem,
            min_size,
            verbose=True,
            block_shape=block_shape,
        )

        # --------------------------------------------------------------
        # 8️⃣  Post‑processing – copy result back into seg_mem
        # --------------------------------------------------------------
        seg_mem[:] = _postprocess_seg_3d(
            seg_mem,
            area_threshold=area_threshold,
            iterations=4,
            iterations_3d=8,
        )

        # --------------------------------------------------------------
        # 9️⃣  If we used a temporary file for the final segmentation,
        #      copy it back into the real dataset.
        # --------------------------------------------------------------
        if tmp_file_for_seg is not None:
            seg_tmp = np.memmap(
                filename=tmp_file_for_seg,
                dtype=np.uint8,
                mode="r",
                shape=foreground.shape,
                order="C",
            )
            seg_ds[:] = seg_tmp[:]
            del seg_tmp
            os.remove(tmp_file_for_seg)

    # --------------------------------------------------------------
    # 10️⃣  **Flush the file** so the reader sees everything.
    # --------------------------------------------------------------
        f_out.flush()
        gc.collect()

        
    # ------------------------------------------------------------------
    # Return the (read‑only) mem‑maps for the intermediates – they can be
    # inspected later if you wish, but the UI does not need them.
    # ------------------------------------------------------------------
    return {"dist": dist, "hmap": hmap, "seeds": seeds}


def run_correction(input_path: str, output_path: str, fname: str) -> bool:
    continue_correction = True

    # ------------------------------------------------------------------
    # 1️⃣  Load the *mandatory* datasets as mem‑maps (tiny RAM footprint)
    # ------------------------------------------------------------------
    raw = _read_h5_memmap(input_path, "data", dtype=np.float32)
    mitos = _read_h5_memmap(input_path, "seg", dtype=np.uint8)
    cristae = _read_h5_memmap(input_path, "labels/cristae", dtype=np.uint8)

    # ------------------------------------------------------------------
    # 2️⃣  Load the *extra* datasets (foreground & boundary) as mem‑maps
    # ------------------------------------------------------------------
    extra_layers = {}
    for key in EXTRA_KEYS:
        data = _read_h5_memmap(input_path, key, dtype=np.float32)
        if data is not None:
            extra_layers[key] = data
        else:
            print(f"Warning: extra key '{key}' not found – skipping.")

    # ------------------------------------------------------------------
    # 3️⃣  Build the Napari viewer – note that we add the *mem‑mapped*
    #     arrays directly; Napari will read them lazily.
    # ------------------------------------------------------------------
    v = napari.Viewer()
    v.add_image(raw, name="raw")
    v.add_labels(mitos, name="mitos")
    if cristae is not None:
        v.add_labels(cristae, name="cristae")

    for key, data in extra_layers.items():
        layer_name = key.split("/")[-1]          # "foreground" or "boundary"
        v.add_image(data, name=layer_name, blending="additive")

    v.title = f"Tomo: {fname}, mitochondria"

    # ------------------------------------------------------------------
    # 4️⃣  The *Run Segmentation* button – now calls the mem‑map version
    # ------------------------------------------------------------------
    @magicgui(
        # … existing widgets …
        recompute={
            "label": "Re‑calculate everything",
            "widget_type": "CheckBox",
            "value": True,
            "tooltip": "Force recomputation of distance‑transform, h‑map and seeds",
        },
        # … keep call_button=False …
    )
    @magicgui(call_button="Run Segmentation")
    def run_segmentation(v: napari.Viewer):
        """Read UI parameters, run the *mem‑mapped* segmentation, and show the result."""
        if "foreground" not in v.layers or "boundary" not in v.layers:
            show_info("Both 'foreground' and 'boundary' layers must be present.")
            return

        foreground = v.layers["foreground"].data
        boundary = v.layers["boundary"].data

        # Pull the current values from the parameter widget (same as before)
        try:
            block_shape = _parse_triplet(param_gui.block_shape.value)
            halo        = _parse_triplet(param_gui.halo.value)
        except ValueError as e:
            show_info(str(e))
            return

        seed_distance      = param_gui.seed_distance.value
        boundary_threshold = param_gui.boundary_threshold.value
        min_size           = param_gui.min_size.value
        area_threshold     = param_gui.area_threshold.value
        recompute          = param_gui.recompute.value

        if "segmentation" in v.layers:
            old_meta = v.layers["segmentation"].metadata
            old_file = old_meta.get("h5_file")
            if old_file is not None:
                try:
                    old_file.close()
                except Exception:
                    pass
                # remove the reference so we don’t try to close it again
                old_meta.pop("h5_file", None)
        
        # ------------------------------------------------------------------
        # 5️⃣  Run the *disk‑backed* segmentation.
        #     The function writes the final label image straight into
        #     ``output_path`` under ``labels/mitochondria``.
        # ------------------------------------------------------------------
        _segment_mitos_mem(
            foreground=foreground,
            boundary=boundary,
            out_path=output_path,
            block_shape=block_shape,
            halo=halo,
            seed_distance=seed_distance,
            boundary_threshold=boundary_threshold,
            min_size=min_size,
            area_threshold=area_threshold,
            recompute=recompute,
        )

        # ------------------------------------------------------------------
        # 6️⃣  Show the newly written segmentation as a *lazy* layer.
        # ------------------------------------------------------------------
        # We open the file read‑only and hand the dataset to Napari
        seg_file = h5py.File(output_path, "r", swmr=True)   # ← SWMR reader
        seg_view = seg_file["labels/mitochondria"]          
        if "segmentation" in v.layers:
            v.layers["segmentation"].data = seg_view
        else:
            v.add_labels(seg_view, name="segmentation")
        v.layers["segmentation"].metadata["h5_file"] = seg_file  # store handle
        show_info("Segmentation finished – layer 'segmentation' added/updated.")

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

    @magicgui(call_button="Paint New Vesicle [f]")
    def paint_new_mitos(v: napari.Viewer):
        """Switch the mitochondria layer to paint mode."""
        layer = v.layers["mitos"]
        layer.selected_label = 1
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
    # 7️⃣  Add the UI widgets (unchanged except for the param_gui line)
    # ------------------------------------------------------------------
    param_gui = seg_params_widget               # no parentheses – keep the FunctionGui
    v.window.add_dock_widget(param_gui.native, area="right", name="Segmentation parameters")
    v.window.add_dock_widget(run_segmentation)


    v.window.add_dock_widget(paint_new_mitos)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(save_correction_cristae)
    v.window.add_dock_widget(save_correction_mitos)
    v.window.add_dock_widget(stop_correction)

    v.bind_key("s", lambda _: save_correction(v))
    v.bind_key("q", lambda _: stop_correction(v))
    v.bind_key("f", lambda _: paint_new_mitos(v))
    v.bind_key("r", lambda _: run_segmentation(v))

    napari.run()
    return continue_correction


def correct_mitochondria(args):
    base_path = args.base_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    file_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))

    iteration = 0
    continue_from = 1
    continue_now = False

    for path in tqdm(file_paths):
        if continue_from == iteration:
            continue_now = True
        iteration += 1
        if not continue_now:
            continue

        _, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        fname = fname.replace("_raw", "") + ".h5"
        output_path = os.path.join(save_dir, fname)

        if os.path.exists(output_path) and not args.force_overwrite:
            print(f"Already exists:\n{output_path}\n")
            continue

        print(f"Loading:\n{path}\nwill save to:\n{output_path}\n")

        if not run_correction(path, output_path, fname):
            break


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
        default=SAVE_DIR,
        help="Path to save the corrected data",
    )
    parser.add_argument(
        "--force_overwrite",
        "-f",
        action="store_true",
        help="Overwrite existing segmentation results",
    )
    args = parser.parse_args()
    correct_mitochondria(args)


if __name__ == "__main__":
    main()