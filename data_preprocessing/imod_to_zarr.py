"""Rasterize IMOD .mod contours into zarr label volumes WITHOUT a raw image stack.

`imodmop` (used by from_mrc_imod_to_h5.py) needs an image of the target size, so this
script paints the closed contours itself:

  1. `imodinfo -a`  -> volume dims (`max`), voxel size, object names
  2. `imodmesh -s -P <gap>` + `imodfillin -e` -> interpolated contours on every slice
     (skipped when the model is already dense)
  3. `model2point -ob` -> obj cont x y z
  4. per z-chunk block: skimage.draw.polygon with IMOD's even-odd nesting rule
     (inner contour = hole), y flipped like the rest of the pipeline

Objects are grouped by name into keys `mitos`, `myelin`, `axons`, `other`; label ids run
1..N per key across all input files. Object table lands in array attrs and a CSV.

Run inside `mamba activate synapse` (IMOD binaries must be on PATH):

    python data_preprocessing/imod_to_zarr.py "a.mod" ["b.mod=Axon Left" ...] -o out.zarr [--suffix _2020] [--no_flip_y]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile

import numcodecs
import numpy as np
import zarr

CHUNKS = (32, 256, 256)  # block z == chunk z, so every chunk is written once
CLASSES = (("mito", "mitos"), ("myelin", "myelin"), ("axon", "axons"))  # order matters


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout


def read_header(mod):
    """-> (max_xyz, voxel_um_zyx, scale_xyz, names). Names come from `imodinfo -a`;
    synapse_net.get_label_names keeps only the last token ("Axon 1" -> "1"), so unusable."""
    mx = scale = pix = units = None
    names = []
    for line in sh(["imodinfo", "-a", mod]).splitlines():
        if line.startswith("max "):
            mx = tuple(int(v) for v in line.split()[1:4])
        elif line.startswith("scale "):
            scale = tuple(float(v) for v in line.split()[1:4])
        elif line.startswith("pixsize"):
            pix = float(line.split()[1])
        elif line.startswith("units"):
            units = line.split()[1]
        elif line.startswith("object "):
            names.append("")
        elif line.startswith("name") and names and not names[-1]:
            names[-1] = line[4:].strip()
    assert mx and scale and pix and units == "nm", f"unexpected header in {mod}: {mx} {scale} {pix} {units}"
    voxel_um = [pix * scale[2] / 1000, pix * scale[1] / 1000, pix * scale[0] / 1000]
    return mx, voxel_um, scale, names


def read_points(mod, tmpdir):
    """-> float array (N, 5): obj cont x y z, obj/cont 1-based, index coords, z = slice index."""
    out = os.path.join(tmpdir, "pts.txt")
    sh(["model2point", "-ob", mod, out])
    return np.fromfile(out, sep=" ").reshape(-1, 5)


def max_z_gap(pts):
    gap = 1
    for o in np.unique(pts[:, 0]):
        z = np.unique(pts[pts[:, 0] == o, 4])
        if len(z) > 1:
            gap = max(gap, int(np.diff(z).max()))
    return gap


def densify(mod, tmpdir, passes):
    """imodmesh across gaps, then imodfillin -e (new contours into existing objects)."""
    src, dst = os.path.join(tmpdir, "in.mod"), os.path.join(tmpdir, "filled.mod")
    shutil.copy(mod, src)
    sh(["imodmesh", "-s", "-P", str(passes), src])
    sh(["imodfillin", "-e", src, dst])
    return dst


def classify(name):
    n = name.lower()
    for kw, key in CLASSES:
        if kw in n:
            return key
    return "other"


def split_contours(pts):
    """-> list of (obj, z, xy[N,2]) for every contour with >= 3 points."""
    brk = np.flatnonzero((np.diff(pts[:, 0]) != 0) | (np.diff(pts[:, 1]) != 0)) + 1
    out, skipped = [], 0
    for c in np.split(pts, brk):
        if len(c) < 3:
            skipped += 1
            continue
        out.append((int(c[0, 0]), int(round(c[0, 4])), c[:, 2:4]))
    if skipped:
        print(f"  skipped {skipped} contours with < 3 points")
    return out


def to_rc(xy, ny, flip_y):
    """IMOD pixel i spans [i, i+1) -> center i+0.5; polygon() tests integer pixel centers."""
    c = xy[:, 0] - 0.5
    r = (ny - 0.5 - xy[:, 1]) if flip_y else (xy[:, 1] - 0.5)
    return r, c


def fill_parity(r, c, shape):
    """Even-odd scanline fill tested at integer pixel centers (IMOD semantics). -> bool mask.
    Pixel (row, col) is inside iff an odd number of edge crossings of the line y=row lie at x <= col.
    skimage.draw.polygon gives the same result but is O(area * vertices): 3 s per big myelin ring."""
    r2, c2 = np.roll(r, -1), np.roll(c, -1)
    y0, y1 = np.ceil(np.minimum(r, r2)).astype(int), np.ceil(np.maximum(r, r2)).astype(int)  # rows [y0, y1)
    n = np.clip(y1 - y0, 0, None)  # horizontal edges -> 0
    e = np.repeat(np.arange(len(r)), n)
    rows = y0[e] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    cols = np.ceil(c[e] + (rows - r[e]) / (r2[e] - r[e]) * (c2[e] - c[e])).astype(int)
    keep = (rows >= 0) & (rows < shape[0])
    acc = np.zeros((shape[0], shape[1] + 1), np.int32)  # extra column swallows crossings right of bbox
    np.add.at(acc, (rows[keep], np.clip(cols[keep], 0, shape[1])), 1)
    return (np.cumsum(acc, axis=1)[:, :-1] & 1).astype(bool)


def rasterize(ds, conts, flip_y):
    """conts: list of (label, z, xy). Paints into zarr array ds. -> voxel count per label."""
    nz, ny, nx = ds.shape
    nlab = max(l for l, _, _ in conts)
    counts = np.zeros(nlab + 1, np.int64)
    cz = np.array([z for _, z, _ in conts])
    for z0 in range(0, nz, ds.chunks[0]):
        z1 = min(z0 + ds.chunks[0], nz)
        idx = np.flatnonzero((cz >= z0) & (cz < z1))
        if not len(idx):
            continue
        rc = [to_rc(conts[i][2], ny, flip_y) for i in idx]
        r0 = max(0, int(np.floor(min(r.min() for r, _ in rc))))
        r1 = min(ny, int(np.ceil(max(r.max() for r, _ in rc))) + 1)
        c0 = max(0, int(np.floor(min(c.min() for _, c in rc))))
        c1 = min(nx, int(np.ceil(max(c.max() for _, c in rc))) + 1)
        blk = np.zeros((z1 - z0, r1 - r0, c1 - c0), np.uint16)
        for i, (r, c) in zip(idx, rc):
            lab, z, _ = conts[i]
            # fill only inside this contour's own bbox
            a0, b0 = max(0, int(np.floor(r.min())) - r0), max(0, int(np.floor(c.min())) - c0)
            a1, b1 = min(blk.shape[1], int(np.ceil(r.max())) + 1 - r0), min(blk.shape[2], int(np.ceil(c.max())) + 1 - c0)
            m = fill_parity(r - r0 - a0, c - c0 - b0, (a1 - a0, b1 - b0))
            sl = blk[z - z0, a0:a1, b0:b1]
            sl[m] = np.where(sl[m] == lab, 0, lab)  # even-odd: nested contour toggles -> hole
            # ponytail: overlaps between different labels of one key = last writer wins, unchecked
        ds[z0:z1, r0:r1, c0:c1] = blk
        counts += np.bincount(blk.ravel(), minlength=nlab + 1)[: nlab + 1]
    return counts


def write_voxel_size(f, ds, voxel_size):
    """Same triple convert_tifstack_to_arbitrary.write_voxel_size writes (not importable: pulls torch)."""
    meta = {"voxel_size": voxel_size, "axes": ["z", "y", "x"], "unit": "micrometer"}
    ds.attrs.update(meta)
    f.attrs.update(meta)


def selftest():
    """Square with a square hole, flipped y; fails if polygon offset / even-odd / flip break."""
    g = zarr.group()
    ds = g.create_dataset("t", shape=(2, 20, 20), chunks=(2, 20, 20), dtype="uint16")
    outer = np.array([[2, 2], [12, 2], [12, 12], [2, 12]], float)
    inner = np.array([[5, 5], [9, 5], [9, 9], [5, 9]], float)
    counts = rasterize(ds, [(1, 0, outer), (1, 0, inner), (2, 1, inner)], flip_y=True)
    a = ds[0]
    assert a[19 - 2, 2] == 1 and a[19 - 11, 11] == 1, "outer square corners (flipped y)"
    assert a[19 - 7, 7] == 0, "hole not cleared"
    assert a[19 - 2, 12] == 0 and a[19 - 12, 2] == 0, "pixel-center offset off by one"
    assert counts[1] == 100 - 16 and counts[2] == 16, counts
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mods", nargs="*", help="path.mod or path.mod=Name (Name overrides all object names in that file)")
    p.add_argument("-o", "--output", help="output .zarr (opened in append mode)")
    p.add_argument("--suffix", default="", help="appended to every key, e.g. _2020")
    p.add_argument("--no_flip_y", action="store_true", help="keep IMOD y (default flips like from_mrc_imod_to_h5)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    assert args.mods and args.output, "need MOD files and -o"
    flip_y = not args.no_flip_y

    shape_xyz = voxel_um = scale = None
    conts, rows, nlab = {}, {}, {}  # per key
    with tempfile.TemporaryDirectory() as tmp:
        for arg in args.mods:
            mod, forced = (arg, None) if os.path.exists(arg) else arg.rsplit("=", 1)
            mx, vox, sc, names = read_header(mod)
            if shape_xyz is None:
                shape_xyz, voxel_um, scale = mx, vox, sc
            assert mx == shape_xyz, f"{mod}: max {mx} != {shape_xyz} of first model"
            if forced:
                names = [forced] * len(names)
            print(f"{os.path.basename(mod)}: max xyz={mx} zscale={sc[2]:g} objects={len(names)}")

            pts = read_points(mod, tmp)
            gap = max_z_gap(pts)
            if gap > 1:
                print(f"  max z-gap {gap} -> imodmesh -s -P {gap + 1} + imodfillin")
                pts = read_points(densify(mod, tmp, gap + 1), tmp)
            assert pts[:, 0].max() <= len(names), "imodfillin created new objects (-e not honoured)"

            obj2lab = {}
            for o in range(1, len(names) + 1):
                zs = np.unique(pts[pts[:, 0] == o, 4]).astype(int)
                if not len(zs):
                    print(f"  object {o} '{names[o - 1]}' has no points, skipped")
                    continue
                key = classify(names[o - 1]) + args.suffix
                lab = nlab[key] = nlab.get(key, 0) + 1
                obj2lab[o] = (key, lab)
                m = re.match(r"\s*(\d+)\s+mito", names[o - 1], re.I)
                rows.setdefault(key, []).append(dict(
                    key=key, label=lab, imod_object=o, file=os.path.basename(mod), name=names[o - 1],
                    parent_axon=int(m.group(1)) if m else "", z_min=int(zs.min()), z_max=int(zs.max()), n_z=len(zs)))
            for o, z, xy in split_contours(pts):
                key, lab = obj2lab[o]
                conts.setdefault(key, []).append((lab, z, xy))

    gaps = [f"{r['name']}({r['n_z']}/{r['z_max'] - r['z_min'] + 1})" for rs in rows.values() for r in rs
            if r["n_z"] < r["z_max"] - r["z_min"] + 1]
    if gaps:
        print(f"WARNING: {len(gaps)} objects still have z gaps: {gaps}")
    for key in set(rows) - set(conts):
        print(f"{key}: only point/line contours (< 3 points), nothing to paint: {[r['name'] for r in rows[key]]}")

    shape = shape_xyz[::-1]
    g = zarr.open_group(args.output, mode="a")
    for k in g.array_keys():
        assert g[k].shape == shape, f"{args.output}/{k} has shape {g[k].shape}, models say {shape}"
    g.attrs.update(imod_max=list(shape_xyz), imod_scale=list(scale), flip_y=flip_y)
    src = dict(g.attrs.get("source_mod", {}))
    for key in conts:
        assert nlab[key] < 65535
        ds = g.create_dataset(key, shape=shape, chunks=CHUNKS, dtype="uint16", fill_value=0, overwrite=True,
                              write_empty_chunks=False,
                              compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE))
        counts = rasterize(ds, conts[key], flip_y)
        for r in rows[key]:
            r["n_voxels"] = int(counts[r["label"]])
        empty = [r["name"] for r in rows[key] if r["n_voxels"] == 0]
        if empty:
            print(f"WARNING: {key}: labels with 0 voxels: {empty}")
        write_voxel_size(g, ds, voxel_um)
        ds.attrs["objects"] = rows[key]
        src[key] = [a.rsplit("=", 1)[0] if not os.path.exists(a) else a for a in args.mods]
        print(f"{key}: {nlab[key]} labels, {int(counts[1:].sum()):,} voxels, {len(conts[key])} contours painted")
    g.attrs["source_mod"] = src

    csv_path = args.output.rstrip("/").removesuffix(".zarr") + "_objects.csv"
    all_rows = [r for k in sorted(g.array_keys()) for r in g[k].attrs.get("objects", [])]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "label", "imod_object", "file", "name", "parent_axon",
                                          "z_min", "z_max", "n_z", "n_voxels"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
