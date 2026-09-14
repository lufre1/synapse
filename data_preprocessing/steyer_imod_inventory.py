"""Inventory of the Steyer FIBSEM IMOD models.

Runs imodinfo over every .mod file in the steyer_data folders, parses the
binary model header (image dims, pixel size), lists all objects with their
full names / contour types / counts, classifies objects into label classes
(mitochondria / axons / myelin / ...), and compares model dims against the
CLAHE tif stacks. Writes a YAML report.

Also serves as the shared registry (DATASETS, parse helpers) for
steyer_imod_to_zarr.py.

Run (synapse env):
    python data_preprocessing/steyer_imod_inventory.py \
        -o /mnt/lustre-grete/usr/u15205/volume-em/steyer_imod_inventory.yaml
"""
import argparse
import os
import re
import struct
import subprocess
import tempfile
from glob import glob

import yaml

IMOD_BIN = "/mnt/lustre-grete/usr/u15205/software/IMOD/bin"

STEYER_ROOT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/orig_files"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u15205/volume-em"
VOXEL_SIZE = (0.025, 0.005, 0.005)  # z, y, x in micrometer

DATASETS = {
    "4005": {"raw_dir": f"{STEYER_ROOT}/2018-08-10_Steyer_Plp_-y_4005/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4005 Plp"},
    "4007": {"raw_dir": f"{STEYER_ROOT}/2019-06-07_Steyer_Plp-4007/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4007 Plp"},
    "4009": {"raw_dir": f"{STEYER_ROOT}/2019-05-14_Steyer_Plp-4009-wt/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4009 wt"},
    "4010": {"raw_dir": f"{STEYER_ROOT}/2019-05-10_Steyer_Plp-4010-wt/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4010 wt"},
    "4015": {"raw_dir": f"{STEYER_ROOT}/2018-08-06_Steyer_Plp_-y_4015/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4015 Plp"},
    "4016": {"raw_dir": f"{STEYER_ROOT}/2018-07-31_Steyer_Plp_wt_4016/CLAHE",
             "mod_dir": f"{STEYER_ROOT}/steyer_data/4016 wt"},
}

# imodinfo header: units code -> name
_UNIT_NAMES = {0: "pixels", 1: "m", -2: "cm", -3: "mm", -6: "um", -9: "nm", -10: "angstrom"}


def imod_env():
    """Environment with the user-space IMOD install on PATH."""
    env = os.environ.copy()
    env["IMOD_DIR"] = os.path.dirname(IMOD_BIN)
    env["PATH"] = IMOD_BIN + os.pathsep + env.get("PATH", "")
    return env


def parse_model_header(mod_path):
    """Read image dims, scales and pixel size straight from the binary .mod header."""
    with open(mod_path, "rb") as f:
        magic = f.read(8)
        assert magic[:4] == b"IMOD", f"{mod_path} is not an IMOD model file"
        f.read(128)  # model name
        fields = struct.unpack(">3iiI4i3f3f3i2ifii3f", f.read(26 * 4))
    xmax, ymax, zmax = fields[0:3]
    zscale = fields[14]
    pixsize = fields[20]
    units = fields[21]
    return {
        "xmax": xmax, "ymax": ymax, "zmax": zmax,
        "zscale": round(float(zscale), 4),
        "pixsize": round(float(pixsize), 4),
        "units": _UNIT_NAMES.get(units, str(units)),
    }


def parse_objects(mod_path):
    """Full object list via imodinfo: id -> {name, type, n_contours}.

    Unlike synapse_net.imod.export.get_label_names this keeps multi-word
    object names ('1 Mito 3') intact.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        subprocess.run(["imodinfo", "-f", tmp.name, mod_path], env=imod_env(),
                       check=True, capture_output=True)
        objects, object_id = {}, None
        with open(tmp.name, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OBJECT"):
                    object_id = int(line.split()[-1])
                    objects[object_id] = {"name": "", "type": "", "n_contours": 0}
                elif object_id is None:
                    continue
                elif line.startswith("NAME:"):
                    objects[object_id]["name"] = line[len("NAME:"):].strip()
                elif "object uses" in line:
                    objects[object_id]["type"] = " ".join(line.split()[2:]).rstrip(".")
                elif re.match(r"^\d+\s+contours", line):
                    objects[object_id]["n_contours"] = int(line.split()[0])
    return objects


def classify_object(name):
    """Map an IMOD object name to a label class. Order matters."""
    low = name.lower()
    if "mito" in low:
        return "mitochondria"
    if "myelin" in low:
        return "myelin"
    if "swell" in low:
        return "swellings"
    if "axon" in low:
        return "axons"
    return "other"


def stack_info(raw_dir):
    """Slice count, xy shape and slice-number gaps of a tif stack."""
    import imageio.v3 as iio
    tifs = sorted(glob(os.path.join(raw_dir, "*.tif")))
    numbers = []
    for t in tifs:
        m = re.search(r"(\d+)\.tif$", os.path.basename(t))
        numbers.append(int(m.group(1)) if m else -1)
    gaps = [(numbers[i - 1], numbers[i]) for i in range(1, len(numbers))
            if numbers[i] != numbers[i - 1] + 1]
    im0 = iio.imread(tifs[0])
    return {
        "n_slices": len(tifs),
        "shape_yx": list(im0.shape),
        "dtype": str(im0.dtype),
        "first_number": numbers[0], "last_number": numbers[-1],
        "numbering_gaps": [list(g) for g in gaps],
    }


def build_inventory(dataset_ids=None):
    inventory = {}
    for ds_id, cfg in DATASETS.items():
        if dataset_ids and ds_id not in dataset_ids:
            continue
        print(f"=== dataset {ds_id}")
        raw = stack_info(cfg["raw_dir"])
        print(f"  raw: {raw['n_slices']} slices of {raw['shape_yx']}, "
              f"numbers {raw['first_number']}-{raw['last_number']}, gaps: {raw['numbering_gaps']}")
        mods = {}
        for mod_path in sorted(glob(os.path.join(cfg["mod_dir"], "*.mod"))):
            stem = os.path.splitext(os.path.basename(mod_path))[0]
            size_mb = os.path.getsize(mod_path) / 1e6
            print(f"  mod: {stem} ({size_mb:.1f} MB)")
            header = parse_model_header(mod_path)
            objects = parse_objects(mod_path)
            class_counts = {}
            for obj in objects.values():
                cls = classify_object(obj["name"])
                class_counts[cls] = class_counts.get(cls, 0) + 1
            dims_match = (header["xmax"] == raw["shape_yx"][1]
                          and header["ymax"] == raw["shape_yx"][0]
                          and header["zmax"] == raw["n_slices"])
            print(f"    dims xyz: {header['xmax']}x{header['ymax']}x{header['zmax']} "
                  f"(match stack: {dims_match}), pixsize {header['pixsize']} {header['units']}, "
                  f"zscale {header['zscale']}")
            print(f"    {len(objects)} objects: {class_counts}")
            mods[stem] = {
                "path": mod_path,
                "size_mb": round(size_mb, 1),
                "header": header,
                "dims_match_stack": bool(dims_match),
                "n_objects": len(objects),
                "class_counts": class_counts,
                "objects": {
                    oid: {**obj, "class": classify_object(obj["name"])}
                    for oid, obj in objects.items()
                },
            }
        inventory[ds_id] = {"raw_dir": cfg["raw_dir"], "raw": raw, "mods": mods}
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=os.path.join(OUTPUT_ROOT, "steyer_imod_inventory.yaml"))
    parser.add_argument("-d", "--datasets", nargs="+", default=None,
                        help="Subset of dataset ids (default: all)")
    args = parser.parse_args()

    inventory = build_inventory(args.datasets)
    with open(args.output, "w") as f:
        yaml.safe_dump(inventory, f, sort_keys=False)
    print(f"\nInventory written to {args.output}")


if __name__ == "__main__":
    main()
