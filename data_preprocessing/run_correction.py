"""
Interactive correction tool for cristae / mitochondria annotations.

Each sample consists of three files sharing a common base name:
  <base>_rec.mrc              — raw tomogram
  <base>_model.tif            — manual annotation to correct (editable Labels layer)
  <base>_rec_prediction.tif   — model prediction (reference Labels layer)

Corrected labels are exported as TIFF to --save_dir.

MRC alignment
-------------
MRC files often use an inverted y-axis relative to TIFF convention, so the raw
image needs to be flipped before it aligns with the segmentation layers.
Use --flip to choose the axis:  y (default), x, xy (180° rotation), or none.

Widgets (right panel)
---------------------
  Relabel layer  — parallel connected components via elf.parallel.label;
                   blockwise CC merged across boundaries, uses all available threads.
  Move instance  — pick an instance ID and move it between any two Labels layers.
                   Source and target layer dropdowns are auto-populated from the viewer.
  Save           — writes every layer to <save_dir>/<raw_basename>_<layer_name>.tif

Key bindings
------------
  s  — Save
  n  — Save & advance to next sample
  q  — Skip (no save)

Usage
-----
  python data_preprocessing/run_correction.py
  python data_preprocessing/run_correction.py -p <data_dir> -o <out_dir> --flip y
"""

import os
from glob import glob

import mrcfile
import napari
import numpy as np
import tifffile
from magicgui import magicgui
from napari.utils.notifications import show_info
from elf.parallel import label as elf_label

DEFAULT_DATA_PATH = "/home/freckmann15/data/cristae/wichmann/2026-05-26"
DEFAULT_SAVE_DIR  = "/home/freckmann15/data/cristae/wichmann/2026-05-26/corrected"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_sample_groups(base_path):
    """Return list of (base, rec_path, model_path, pred_path) for every complete sample."""
    rec_files = sorted(glob(os.path.join(base_path, "*_rec.mrc")))
    groups = []
    for rec in rec_files:
        base  = rec[:-len("_rec.mrc")]
        model = base + "_model.tif"
        pred  = base + "_rec_prediction.tif"
        if os.path.exists(model) and os.path.exists(pred):
            groups.append((base, rec, model, pred))
        else:
            print(f"  WARNING: incomplete group for {os.path.basename(base)}, skipping.")
    return groups


# ---------------------------------------------------------------------------
# Napari session for one sample
# ---------------------------------------------------------------------------

def _flip_raw(raw, flip):
    """Flip raw MRC array to match TIFF axis convention."""
    if flip == "y":
        return np.flip(raw, axis=1)
    if flip == "x":
        return np.flip(raw, axis=2)
    if flip == "xy":
        return np.flip(raw, axis=(1, 2))   # 180° rotation in YX plane
    return raw   # "none"


def run_correction(base_name, rec_path, model_path, pred_path, save_dir, args):
    """Open napari for one sample. Returns True to continue, False to stop all."""
    continue_session = [True]   # list so closures can mutate it
    fname = os.path.basename(base_name)
    already_saved = any(
        f.startswith(fname + "_") and f.endswith(".tif")
        for f in os.listdir(save_dir)
    ) if os.path.isdir(save_dir) else False

    # --- load data ---
    with mrcfile.open(rec_path, "r") as f:
        raw = np.array(f.data, dtype=np.float32)
    raw = _flip_raw(raw, args.flip)
    model      = tifffile.imread(model_path).astype(np.int32)
    prediction = tifffile.imread(pred_path).astype(np.int32)

    # --- build viewer ---
    viewer = napari.Viewer()
    viewer.title = f"{'[saved] ' if already_saved else ''}[{fname}]"

    viewer.add_image(raw, name="raw")
    viewer.add_labels(prediction, name="cristae", opacity=0.55)
    edit_layer = viewer.add_labels(model, name="mitochondria")
    edit_layer.mode = "paint"

    # --- helpers ---
    def _save():
        os.makedirs(save_dir, exist_ok=True)
        saved = []
        for layer in viewer.layers:
            safe_name = layer.name.replace(" ", "_").replace("—", "-").replace("/", "-")
            out_path = os.path.join(save_dir, f"{fname}_{safe_name}.tif")
            data = layer.data
            if isinstance(layer, napari.layers.Labels):
                tifffile.imwrite(out_path, data.astype(np.int32), compression="zlib")
            else:
                tifffile.imwrite(out_path, data, compression="zlib")
            saved.append(os.path.basename(out_path))
        viewer.title = f"[saved] [{fname}]"
        show_info(f"Saved {len(saved)} layer(s) → {save_dir}")

    # --- widgets ---
    @magicgui(call_button="Save  [s]")
    def btn_save():
        _save()

    @magicgui(call_button="Save & Next  [n]")
    def btn_save_next():
        _save()
        viewer.close()

    @magicgui(call_button="Skip — no save  [q]")
    def btn_skip():
        viewer.close()

    @magicgui(call_button="Stop all")
    def btn_stop():
        continue_session[0] = False
        viewer.close()

    @magicgui(
        call_button="Move instance",
        instance_id={"widget_type": "SpinBox", "label": "Instance ID", "min": 0, "max": 2**31 - 1, "value": 1},
    )
    def btn_move_instance(
        source_layer: napari.layers.Labels,
        target_layer: napari.layers.Labels,
        instance_id: int = 1,
    ):
        if source_layer is None or target_layer is None:
            show_info("Select both source and target layers.")
            return
        if source_layer.name == target_layer.name:
            show_info("Source and target must be different layers.")
            return
        mask = source_layer.data == instance_id
        if not np.any(mask):
            show_info(f"Instance {instance_id} not found in '{source_layer.name}'.")
            return
        target_layer.data[mask] = instance_id
        source_layer.data[mask] = 0
        source_layer.refresh()
        target_layer.refresh()
        show_info(
            f"Moved instance {instance_id}: '{source_layer.name}' → '{target_layer.name}'"
        )

    @magicgui(call_button="Relabel layer (CC)")
    def btn_relabel(layer: napari.layers.Labels):
        if layer is None:
            show_info("Select a layer to relabel.")
            return
        n_before = len(np.unique(layer.data)) - 1
        relabeled = elf_label(layer.data > 0, with_background=True, block_shape=(64, 512, 512)).astype(np.int32)
        layer.data = relabeled
        layer.refresh()
        n_after = len(np.unique(relabeled)) - 1
        show_info(f"Relabeled '{layer.name}': {n_before} → {n_after} instances")

    for widget in (btn_relabel, btn_move_instance, btn_save, btn_save_next, btn_skip, btn_stop):
        viewer.window.add_dock_widget(widget, area="right")

    # key bindings
    viewer.bind_key("s", lambda _: _save())
    viewer.bind_key("n", lambda _: (_save(), viewer.close()))
    viewer.bind_key("q", lambda _: viewer.close())

    napari.run()
    return continue_session[0]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def correct_samples(args):
    groups = find_sample_groups(args.base_path)
    if not groups:
        print(f"No complete sample groups found in:\n  {args.base_path}")
        return

    print(f"Found {len(groups)} sample(s).  Output → {args.save_dir}\n")
    os.makedirs(args.save_dir, exist_ok=True)

    for i, (base, rec, model, pred) in enumerate(groups):
        fname = os.path.basename(base)
        already_done = os.path.isdir(args.save_dir) and any(
            f.startswith(fname + "_") and f.endswith(".tif")
            for f in os.listdir(args.save_dir)
        )

        if args.skip_done and already_done:
            print(f"  [{i+1}/{len(groups)}] [done] {fname}  — skipping (already saved)")
            continue

        print(f"  [{i+1}/{len(groups)}] {fname}")
        if not run_correction(base, rec, model, pred, args.save_dir, args):
            print("Stopped.")
            break

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive napari correction for cristae / mito annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_path", "-p", default=DEFAULT_DATA_PATH,
                        help="Directory containing *_rec.mrc / *_model.tif / *_rec_prediction.tif triplets")
    parser.add_argument("--save_dir",  "-o", default=DEFAULT_SAVE_DIR,
                        help="Output directory for corrected TIFF files")
    parser.add_argument("--flip", default="y", choices=["y", "x", "xy", "none"],
                        help="Axis to flip the raw MRC along to align with TIFFs. "
                             "'y' flips top-bottom (most common), 'x' flips left-right, "
                             "'xy' rotates 180° in plane, 'none' disables flipping.")
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false",
                        help="Re-open samples that already have a corrected file")
    parser.set_defaults(skip_done=True)
    args = parser.parse_args()
    correct_samples(args)


if __name__ == "__main__":
    main()
