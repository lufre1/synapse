"""Export Steyer FIBSEM datasets to one zarr per dataset: raw multiscale + IMOD labels.

Per dataset this pipeline
  1. converts the CLAHE tif stack to a temporary zarr v2 (key s0) via
     convert_tifstack_to_arbitrary.py,
  2. adds downscaled levels s1-s3 via downscale_zarr.py,
  3. exports every .mod file's objects as instance labels
     (labels/<mod-stem>/<class>, class in mitochondria/axons/myelin/...) using
     imodinfo -a (strip 'nodraw' so switched-off objects paint) -> imodmesh -s
     -> imodfillin -e (interpolate contours across z-gaps) -> imodmop -label /
     (paint each object with its IMOD object number),
  4. writes QC overlay PNGs,
  5. converts the temp store to sharded zarr v3 with the zarr-converters
     project and verifies it.

Instance ids equal the IMOD object numbers; the id -> object-name map is
stored in the label dataset attrs. The mask MRC is y-flipped into numpy/tif
orientation (verified against raw: labels sit on dark structures).
Mods whose geometry does not match the tif stack are skipped, except for the
"annotated stack had extra slices that were later removed" case (4007), which
is handled by a filename-derived z-remap.

Run (synapse env):
    python data_preprocessing/steyer_imod_to_zarr.py -d 4009
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from glob import glob

import imageio.v3 as iio
import mrcfile
import numcodecs
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steyer_imod_inventory import (  # noqa: E402
    DATASETS, OUTPUT_ROOT, classify_object, imod_env, parse_model_header, parse_objects,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZARR_CONVERTERS = "/mnt/vast-nhr/home/freckmann15/u15205/zarr-converters"
ZARR_CONVERTERS_PY = "/mnt/lustre-grete/usr/u15205/envs/zarr-converters/bin/python"

Z_CHUNK = 64
LABEL_CHUNKS = (64, 128, 128)
MESH_PASSES = 30


def run_imod(cmd, **kwargs):
    res = subprocess.run(cmd, env=imod_env(), capture_output=True, text=True, **kwargs)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return res


def dataset_voxel_size(mod_dir):
    """Voxel size (z, y, x) in micrometer from the models' pixsize/zscale (5nm xy; z varies: 4015 is 10nm)."""
    for mod_path in sorted(glob(os.path.join(mod_dir, "*.mod"))):
        h = parse_model_header(mod_path)
        if h["units"] == "nm" and h["pixsize"] > 0:
            return (h["pixsize"] * h["zscale"] / 1000., h["pixsize"] / 1000., h["pixsize"] / 1000.)
    return (0.025, 0.005, 0.005)


def build_z_map(raw_dir, model_nz, stack_nz):
    """Map model z indices -> stack z indices.

    Identity if the dims agree. If the model spans the full tif numbering
    range (annotated before slices went missing, e.g. 4007 lost slice 0362),
    map by filename number and drop missing slices. Returns None if the model
    geometry cannot be related to the stack.
    """
    if model_nz == stack_nz:
        return np.arange(stack_nz)
    import re
    tifs = sorted(glob(os.path.join(raw_dir, "*.tif")))
    numbers = [int(re.search(r"(\d+)\.tif$", os.path.basename(t)).group(1)) for t in tifs]
    span = numbers[-1] - numbers[0] + 1
    if model_nz == span:
        return np.array([n - numbers[0] for n in numbers])  # model z of each stack slice
    return None


def prepare_model(mod_path, work_dir, object_ids):
    """Ascii-dump the model, strip 'nodraw', mesh and fill z-gaps.

    Returns the path of the paint-ready model.
    """
    stem = os.path.splitext(os.path.basename(mod_path))[0]
    ascii_path = os.path.join(work_dir, f"{stem}_on.mod")
    with open(ascii_path, "w") as f:
        res = subprocess.run(["imodinfo", "-a", mod_path], env=imod_env(),
                             stdout=subprocess.PIPE, text=True, check=True)
        started = False
        for line in res.stdout.splitlines(keepends=True):
            if not started and line.startswith("imod "):
                started = True
            if started and line.strip() != "nodraw":
                f.write(line)
    meshed_path = os.path.join(work_dir, f"{stem}_meshed.mod")
    shutil.copy(ascii_path, meshed_path)
    ids = ",".join(str(i) for i in sorted(object_ids))
    run_imod(["imodmesh", "-s", "-P", str(MESH_PASSES), "-C", "-o", ids, meshed_path])
    filled_path = os.path.join(work_dir, f"{stem}_filled.mod")
    run_imod(["imodfillin", "-e", "-o", ids, meshed_path, filled_path])
    os.remove(ascii_path)
    os.remove(meshed_path)
    return filled_path


def paint_class(model_path, ref_mrc, out_mrc, object_ids):
    """imodmop: paint the given objects, each with its object number as value."""
    ids = ",".join(str(i) for i in sorted(object_ids))
    mode = "0" if max(object_ids) <= 127 else "6"
    run_imod(["imodmop", "-mode", mode, "-label", "/", "-objects", ids,
              model_path, ref_mrc, out_mrc])


def point_annotations(model_path, object_ids, ny, z_map):
    """Coordinates of objects made of isolated points (markers, not areas).

    imodmop paints areas, so such objects rasterize to nothing. Return their
    points in the output array's index convention (z, y, x) so the annotation
    is preserved in the dataset attrs instead of being lost.
    """
    with tempfile.NamedTemporaryFile(suffix=".pts") as tmp:
        run_imod(["model2point", "-object", model_path, tmp.name])
        per_contour, points = {}, []
        with open(tmp.name) as f:
            for line in f:
                oid, cid, x, y, z = line.split()
                oid = int(oid)
                if oid not in object_ids:
                    continue
                per_contour[(oid, int(cid))] = per_contour.get((oid, int(cid)), 0) + 1
                points.append((oid, int(cid), float(x), float(y), float(z)))
    if not per_contour or max(per_contour.values()) > 2:
        return None  # real contours: imodmop handles these
    z_index = {int(round(zm)): i for i, zm in enumerate(z_map)}
    out = []
    for oid, _, x, y, z in points:
        zi = z_index.get(int(round(z)))
        if zi is not None:
            out.append([zi, ny - 1 - int(round(y)), int(round(x))])
    return sorted(out)


def stream_mask_to_zarr(mask_mrc, group, key, z_map, id_to_name, meta):
    """Stream the painted MRC into the zarr store: y-flip, z-remap, uint16."""
    with mrcfile.mmap(mask_mrc, permissive=True) as f:
        data = f.data
        out_shape = (len(z_map), data.shape[1], data.shape[2])
        compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        ds = group.require_dataset(key, shape=out_shape, chunks=LABEL_CHUNKS,
                                   dtype="uint16", compressor=compressor)
        per_slice = np.zeros(out_shape[0], dtype=np.int64)
        for start in range(0, out_shape[0], Z_CHUNK):
            end = min(start + Z_CHUNK, out_shape[0])
            block = np.asarray(data[z_map[start:end]]).astype(np.uint16)
            block = np.flip(block, axis=1)
            ds[start:end] = block
            per_slice[start:end] = (block > 0).sum(axis=(1, 2))
    ds.attrs.update({"instance_id_to_name": {str(k): v for k, v in id_to_name.items()}, **meta})
    return ds, per_slice


def export_labels(ds_id, cfg, tmp_zarr, work_dir, voxel_size):
    root = zarr.open(tmp_zarr, mode="a")
    s0 = root["s0"]
    stack_nz, ny, nx = s0.shape

    ref_mrc = os.path.join(work_dir, f"ref_{ds_id}.mrc")
    ref_nz = stack_nz  # may grow to model space below

    report = {}
    qc_slices = {}
    for mod_path in sorted(glob(os.path.join(cfg["mod_dir"], "*.mod"))):
        stem = os.path.splitext(os.path.basename(mod_path))[0]
        header = parse_model_header(mod_path)
        z_map = build_z_map(cfg["raw_dir"], header["zmax"], stack_nz)
        if header["xmax"] != nx or header["ymax"] != ny or z_map is None:
            print(f"  [skip] {stem}: model dims {header['xmax']}x{header['ymax']}x{header['zmax']} "
                  f"do not match stack {nx}x{ny}x{stack_nz}")
            report[stem] = {"status": "skipped_geometry_mismatch", "header": header}
            continue

        objects = parse_objects(mod_path)
        classes = {}
        for oid, obj in objects.items():
            if obj["n_contours"] == 0:
                continue
            classes.setdefault(classify_object(obj["name"]), {})[oid] = obj["name"]
        already = [cls for cls in classes if f"labels/{stem}/{cls}" in root]
        if already:
            for cls in already:
                del classes[cls]
            print(f"  [mod] {stem}: {already} already exported, skipping those")
            report[stem] = {"status": "exported", "classes": {c: "pre-existing" for c in already}}
        if not classes and already:
            continue
        if not classes:
            print(f"  [skip] {stem}: no non-empty contour objects")
            report[stem] = {"status": "skipped_empty"}
            continue

        # Reference MRC in model z-space (>= stack z-space for the 4007 case).
        if header["zmax"] != ref_nz or not os.path.exists(ref_mrc):
            ref_nz = header["zmax"]
            if os.path.exists(ref_mrc):
                os.remove(ref_mrc)
            m = mrcfile.new_mmap(ref_mrc, shape=(ref_nz, ny, nx), mrc_mode=0, overwrite=True)
            m.voxel_size = (1., 1., 1.)  # unit spacing: model coords map 1:1 to voxel indices
            m.close()

        all_ids = [oid for cls_ids in classes.values() for oid in cls_ids]
        print(f"  [mod] {stem}: {len(all_ids)} objects -> {sorted(classes)}")
        model_path = prepare_model(mod_path, work_dir, all_ids)

        report.setdefault(stem, {"status": "exported", "classes": {}})
        for cls, id_to_name in sorted(classes.items()):
            out_mrc = os.path.join(work_dir, f"{ds_id}_{stem}_{cls}.mrc")
            paint_class(model_path, ref_mrc, out_mrc, list(id_to_name))
            group = root.require_group("labels").require_group(stem)
            meta = {
                "class": cls, "source_mod": mod_path,
                "voxel_size": list(voxel_size), "axes": ["z", "y", "x"], "unit": "micrometer",
                "export_date": str(date.today()),
                "note": "instance ids are IMOD object numbers",
            }
            ds, per_slice = stream_mask_to_zarr(out_mrc, group, cls, z_map, id_to_name, meta)
            os.remove(out_mrc)
            n_inst = len(id_to_name)
            print(f"    labels/{stem}/{cls}: {n_inst} instances, "
                  f"{int(per_slice.sum())} voxels")
            if per_slice.sum() == 0:
                # Objects built from isolated points paint no area - keep the
                # coordinates in the attrs rather than losing the annotation.
                pts = point_annotations(model_path, set(id_to_name), ny, z_map)
                if pts:
                    ds.attrs["point_annotations_zyx"] = pts
                    ds.attrs["point_annotations_note"] = (
                        "Objects consist of isolated point markers, not closed areas, so the "
                        "voxel array is empty by design. Coordinates are (z, y, x) in this "
                        "array's index convention, y already flipped to match the raw data.")
                    print(f"      -> point-only object: stored {len(pts)} marker coordinates in attrs")
            report[stem]["classes"][cls] = {"n_instances": n_inst, "n_voxels": int(per_slice.sum())}
            qc_slices[(stem, cls)] = np.argsort(per_slice)[-3:][::-1].tolist()
        os.remove(model_path)

    if os.path.exists(ref_mrc):
        os.remove(ref_mrc)
    return report, qc_slices


def write_qc(tmp_zarr, qc_dir, qc_slices):
    from skimage.segmentation import find_boundaries
    os.makedirs(qc_dir, exist_ok=True)
    root = zarr.open(tmp_zarr, mode="r")
    for (stem, cls), z_list in qc_slices.items():
        lab = root[f"labels/{stem}/{cls}"]
        for z in z_list:
            raw = root["s0"][z]
            seg = lab[z]
            if not seg.any():
                continue
            rgb = np.stack([raw] * 3, -1)
            b = find_boundaries(seg, mode="thick")
            rgb[b] = [255, 40, 40]
            inner = (seg > 0) & ~b
            rgb[inner] = (0.65 * rgb[inner] + 0.35 * np.array([255, 220, 0])).astype(np.uint8)
            out = os.path.join(qc_dir, f"{stem}__{cls}__z{z:04d}.png")
            iio.imwrite(out, rgb[::4, ::4])
    print(f"  QC overlays in {qc_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scratch", default=os.environ.get("STEYER_SCRATCH", "/mnt/lustre-grete/tmp/u15205/steyer_export"))
    parser.add_argument("--keep_tmp", action="store_true", help="Keep the temp v2 store after conversion")
    parser.add_argument("--stop_after", choices=["raw", "labels", "qc"], default=None,
                        help="Stop after the given stage (for piloting)")
    args = parser.parse_args()

    ds_id = args.dataset
    cfg = DATASETS[ds_id]
    final_dir = os.path.join(OUTPUT_ROOT, ds_id)
    final_zarr = os.path.join(final_dir, f"{ds_id}.zarr")
    if os.path.exists(final_zarr):
        raise SystemExit(f"{final_zarr} already exists - refusing to overwrite. Remove it manually if intended.")

    work_dir = os.path.join(args.scratch, ds_id)
    os.makedirs(work_dir, exist_ok=True)
    tmp_zarr = os.path.join(work_dir, f"{ds_id}.zarr")
    voxel_size = dataset_voxel_size(cfg["mod_dir"])
    print(f"=== {ds_id}: voxel size (z,y,x) = {voxel_size} um, scratch = {work_dir}")

    # 1. raw -> s0
    if not os.path.exists(os.path.join(tmp_zarr, "s0")):
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "convert_tifstack_to_arbitrary.py"),
                        "--input_dir", cfg["raw_dir"], "--output_path", tmp_zarr,
                        "--dataset_name", "s0", "--n_threads", "8",
                        "--voxel_size", *[str(v) for v in voxel_size]], check=True)
    # 2. multiscale s1-s3
    for in_key, out_key, scale in [("s0", "s1", ["1", "0.5", "0.5"]),
                                   ("s1", "s2", ["1", "0.5", "0.5"]),
                                   ("s2", "s3", ["0.5", "0.5", "0.5"])]:
        if not os.path.exists(os.path.join(tmp_zarr, out_key)):
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "downscale_zarr.py"),
                            "-i", tmp_zarr, "-k", in_key, "-ok", out_key, "-s", *scale], check=True)
    if args.stop_after == "raw":
        return

    # 3. labels from all mods
    report, qc_slices = export_labels(ds_id, cfg, tmp_zarr, work_dir, voxel_size)
    root = zarr.open(tmp_zarr, mode="a")
    root.attrs.update({"dataset": ds_id, "voxel_size": list(voxel_size),
                       "axes": ["z", "y", "x"], "unit": "micrometer",
                       "raw_source": cfg["raw_dir"], "mod_source": cfg["mod_dir"]})
    if args.stop_after == "labels":
        print(json.dumps(report, indent=2))
        return

    # 4. QC overlays (written straight to the final dataset folder)
    os.makedirs(final_dir, exist_ok=True)
    write_qc(tmp_zarr, os.path.join(final_dir, "qc"), qc_slices)
    with open(os.path.join(final_dir, "qc", "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if args.stop_after == "qc":
        return

    # 5. shard to zarr v3 + verify
    subprocess.run([ZARR_CONVERTERS_PY, os.path.join(ZARR_CONVERTERS, "scripts", "convert_to_v3.py"),
                    tmp_zarr, final_zarr, "--shard-factor", "4"], check=True)
    subprocess.run([ZARR_CONVERTERS_PY, os.path.join(ZARR_CONVERTERS, "scripts", "verify_conversion.py"),
                    tmp_zarr, final_zarr], check=True)

    if not args.keep_tmp:
        shutil.rmtree(tmp_zarr)
    print(f"=== {ds_id} done: {final_zarr}")


if __name__ == "__main__":
    main()
