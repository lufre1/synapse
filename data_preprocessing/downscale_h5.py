import gc
import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
from synapse.h5_util import get_all_keys_from_h5, read_voxel_size
from skimage.transform import rescale


def get_filename_and_inter_dirs(file_path, base_path):
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    relative_path = os.path.relpath(file_path, base_path)
    inter_dirs = os.path.dirname(relative_path)
    return file_name, inter_dirs


def create_directories_if_not_exists(base_path, inter_dirs):
    full_path = os.path.join(base_path, inter_dirs)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"\nCreated directories: {full_path}")
    else:
        print(f"\nDirectories already exist: {full_path}")


def _write_dataset_chunked(f_in, f_out, key, scale_factors, anti_alias, voxel_size_arr, z_chunk=8):
    """Pre-allocate output dataset and fill it z-slab by z-slab to avoid OOM."""
    in_ds = f_in[key]
    in_shape = in_ds.shape
    orig_dtype = np.uint8 if in_ds.dtype == np.int8 else in_ds.dtype
    sf_z, sf_y, sf_x = int(scale_factors[0]), int(scale_factors[1]), int(scale_factors[2])
    scale = (1.0 / sf_z, 1.0 / sf_y, 1.0 / sf_x)
    # Match skimage's own rounding (np.round) so the pre-allocated shape is consistent
    out_shape = (
        max(1, in_shape[0] // sf_z),
        max(1, int(np.round(in_shape[1] * scale[1]))),
        max(1, int(np.round(in_shape[2] * scale[2]))),
    )
    print(f"  {key}: {in_shape} ({orig_dtype}) -> {out_shape}")
    ds_out = f_out.create_dataset(key, shape=out_shape, dtype=orig_dtype, compression="gzip")
    if voxel_size_arr is not None and "raw" in key:
        ds_out.attrs.create("voxel_size", data=voxel_size_arr)
        ds_out.attrs.create("voxel_size_order", data="z, y, x")

    out_y, out_x = out_shape[1], out_shape[2]
    for z_out in range(0, out_shape[0], z_chunk):
        z_out_end = min(z_out + z_chunk, out_shape[0])
        slab = np.array(in_ds[z_out * sf_z: z_out_end * sf_z])
        if anti_alias:
            scaled = rescale(slab.astype(np.float32), scale=scale, order=3,
                             anti_aliasing=True, preserve_range=True).astype(orig_dtype)
        else:
            scaled = slab[::sf_z, ::sf_y, ::sf_x].astype(orig_dtype)
        # Clip to pre-allocated shape in case of rounding edge cases
        ds_out[z_out:z_out_end] = scaled[:, :out_y, :out_x]
        del slab, scaled
        gc.collect()

    return out_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b", type=str,
                        default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi",
                        help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str,
                        default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/s2/",
                        help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, nargs='+', default=[2, 2, 2],
                        help="Scale factor per dimension (z y x)")
    parser.add_argument("--downsample", "-d", action='store_true', default=False,
                        help="Nearest-neighbour subsample (quicker, no anti-aliasing on raw)")
    parser.add_argument("--original_voxel_size", "-ovs", nargs=3, type=float, default=None,
                        help="Fallback voxel size (z y x) if not in file metadata")
    parser.add_argument("--new_voxel_size", "-nvs", nargs=3, type=float, default=None,
                        help="Override output voxel size (z y x)")
    args = parser.parse_args()

    orig_voxel_size = None
    if args.original_voxel_size is not None:
        orig_voxel_size = np.array(args.original_voxel_size, dtype=np.float32)

    if os.path.isfile(args.base_path):
        print("base path is a file:", args.base_path)
        h5_paths = [args.base_path]
    else:
        h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True))

    for path in tqdm(h5_paths):
        export_file_name, rel_path = get_filename_and_inter_dirs(path, args.base_path)
        if os.path.isfile(args.base_path):
            if os.path.splitext(args.export_path)[1]:
                export_file_path = args.export_path
            else:
                export_file_path = os.path.join(args.export_path, rel_path, export_file_name + ".h5")
        else:
            export_file_path = os.path.join(args.export_path, rel_path, export_file_name + ".h5")
            create_directories_if_not_exists(args.export_path, rel_path)

        if os.path.exists(export_file_path):
            print("File already exists:", export_file_path)
            continue

        keys = get_all_keys_from_h5(path)

        voxel_size = None
        for key in keys:
            if "raw" in key:
                voxel_size = read_voxel_size(h5_path=path, h5_key=key, default=None)
                if voxel_size is not None:
                    break
        if voxel_size is None:
            voxel_size = read_voxel_size(h5_path=path, h5_key="raw", default=orig_voxel_size)
        if voxel_size is None:
            print(f"WARNING: no voxel_size metadata found in {path}; voxel size will not be written.")
        else:
            print(f"voxel_size read from metadata: {voxel_size}")

        if voxel_size is not None:
            new_voxel_size = np.asarray(voxel_size, dtype=np.float64) * np.asarray(args.scale_factor)
        else:
            new_voxel_size = None
        if args.new_voxel_size is not None:
            new_voxel_size = np.array(args.new_voxel_size, dtype=np.float32)
        if new_voxel_size is not None:
            print(f"new voxel size: {new_voxel_size}")

        voxel_size_arr = np.array(new_voxel_size, dtype=np.float32) if new_voxel_size is not None else None

        with h5py.File(path, "r") as f_in, h5py.File(export_file_path, "w") as f_out:
            if voxel_size_arr is not None:
                f_out.attrs.create("voxel_size", data=voxel_size_arr)
                f_out.attrs.create("voxel_size_order", data="z, y, x")

            new_shape = None
            for key in keys:
                is_raw = "raw" in key
                anti_alias = is_raw and not args.downsample
                out_shape = _write_dataset_chunked(
                    f_in, f_out, key, args.scale_factor, anti_alias, voxel_size_arr
                )
                if is_raw:
                    new_shape = out_shape

        print(f"new shape: {new_shape}")
        print(f"Data successfully exported to {export_file_path}")


if __name__ == "__main__":
    main()
