import argparse
import os
from glob import glob

import yaml
import h5py
import zarr
import torch
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
import synapse.io.util as io
import synapse.util as util
from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.util import get_prediction
# from synapse_net.ground_truth.matching import find_additional_objects
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.measure import label
from skimage.transform import resize


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", type=str, default=None, help="Path to YAML/JSON config file")

    p.add_argument("--base_path", "-b", type=str)
    p.add_argument("--file_extension", "-fe", type=str)
    p.add_argument("--key", "-k", type=str)
    p.add_argument("--label_path", "-lp", type=str)
    p.add_argument("--label_key", "-lk", type=str)
    p.add_argument("--export_path", "-e", type=str)
    p.add_argument("--model_path", "-m", type=str, required=False)
    p.add_argument("--add_missing_mitos", "-am", action="store_true")
    p.add_argument("--seed_distance", "-sd", type=int)
    p.add_argument("--boundary_threshold", "-bt", type=float)
    p.add_argument("--foreground_threshold", "-ft", type=float)
    p.add_argument("--area_threshold", "-at", type=int)
    p.add_argument("--min_size", "-ms", type=int)
    p.add_argument("--post_iter3d", "-p3d", type=int)
    p.add_argument("--use_custom_segment", "-uc", action="store_true")
    p.add_argument("--tile_shape", "-ts", type=int, nargs=3)
    p.add_argument("--all_keys", "-ak", action="store_true")
    p.add_argument("--force_overwrite", "-fo", action="store_true")
    p.add_argument("--centered_crop", "-cc", action="store_true")
    p.add_argument("--downscale_export", "-de", type=int)
    p.add_argument("--preprocess_volem", "-pv", action="store_true")
    p.add_argument("--disk_based_prediction", "-dbp", action="store_true")
    p.add_argument("--only_foreground", "-of", action="store_true", default=False, help="No boundary predictions")
    p.add_argument("--tmp_folder", "-tmp", type=str, default=None)
    return p

def parse_args():
    parser = build_parser()

    # parse only --config first
    cfg_args, remaining = parser.parse_known_args()

    # load config
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)

    # now parse full args, CLI overrides config defaults
    args = parser.parse_args(remaining)

    # enforce required args after config merge
    if args.model_path is None:
        parser.error("--model_path/-m is required (either in config or CLI).")

    return args

def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def _read_h5(path, key, scale_factor, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == "prediction" or "pred" in key:
                image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            else:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            print(f"{key} data shape after downsampling", image.shape)
            # if not key == "raw":
            #     print(np.unique(image))

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image

def _get_raw_transform(x):
    x = util.convert_white_patches_to_black(x, min_patch_size=100)
    x = torch_em.transform.raw.normalize_percentile(x)
    return x


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys


def get_all_dataset_keys(file_path):
    """
    Returns a list of all dataset keys in a file (HDF5, Zarr, or N5).
    
    Parameters:
        file_path (str): Path to the file or directory.
        
    Returns:
        keys (list): List of dataset keys (paths).
    """
    keys = []

    if os.path.isfile(file_path) and file_path.endswith(('.h5', '.hdf5')):
        # HDF5
        with h5py.File(file_path, 'r') as h5file:
            def collect_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    keys.append(name)
            h5file.visititems(collect_keys)

    else:
        # Assume Zarr or N5 directory
        store = zarr.N5Store(file_path) if 'attributes.json' in os.listdir(file_path) else zarr.DirectoryStore(file_path)
        root = zarr.open(store, mode='r')

        def collect_keys(name, obj):
            if isinstance(obj, zarr.core.Array):
                keys.append(name)
        root.visititems(collect_keys)

    return keys


def main():

    args = parse_args()
    exp_scale = args.downscale_export
    print(args.base_path)
    print("\nUsing model", args.model_path)
    # tile_shape
    z, y, x = args.tile_shape
    ts = {
        "z": z,
        "y": y,
        "x": x
        }
    halo = {
        "z": int(ts["z"] * 0.125),
        "y": int(ts["y"] * 0.125),
        "x": int(ts["x"] * 0.125)
        }
    # if args.use_custom_segment:
    #     # adjust for blocking in torch_em 
    #     ts = {
    #         "z": int(z - 2 * halo["z"]),
    #         "y": int(y - 2 * halo["y"]),
    #         "x": int(x - 2 * halo["x"])
    #         }
    # halo = {'z': 12, 'y': 128, 'x': 128}
    # ts = {'z': ts["z"]+2*halo["z"], 'y': ts["y"]+2*halo["y"], 'x': ts["x"]+2*halo["x"]}
    h5_paths = io.load_file_paths(args.base_path, args.file_extension)

    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": ts, "halo": halo}  # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = None
    bt_string = str(args.boundary_threshold).replace(".", "")
    ft_string = str(args.foreground_threshold).replace(".", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using best model from", args.model_path, "with device", device)

    for path in tqdm(h5_paths):

        print("\nopening file", path)
        if ".zarr" in args.export_path:
            output_path = args.export_path
        else:
            os.makedirs(args.export_path, exist_ok=True)
            output_path = os.path.join(args.export_path, (os.path.basename(args.model_path)).replace(".pt", "") +
                                    f"_sd{args.seed_distance}_bt{bt_string}_ft{ft_string}_with_pred_ts_z{ts['z']}_y{ts['y']}_x{ts['x']}_halo_z{halo['z']}_y{halo['y']}_x{halo['x']}_" +
                                    os.path.basename(path))
            output_path = output_path.replace(".zarr", ".h5")
            if os.path.exists(output_path) and not args.force_overwrite:
                print("Skipping... output path exists", output_path)
                continue
            elif os.path.exists(output_path) and args.force_overwrite:
                print("Overwriting... output path exists", output_path)
                os.remove(output_path)
        if path.endswith(".h5"):
            keys = get_all_keys_from_h5(path)
        else:
            keys = get_all_dataset_keys(path)
        data = {}
        scale_factor = 1
        with open_file(path, "r") as f:
            centered_crop = args.centered_crop
            if args.key is not None and not args.all_keys:
                image = f[args.key]  # [::scale_factor, ::scale_factor, ::scale_factor]
            else:
                image = None
                max_shape = (200, 2000, 2000)  # to not crash
                print("Cropping to", max_shape, "if necessary")
                slices = None
                for idx, key in enumerate(keys):
                    arr = f[key][...]
                    if slices is None:
                        if centered_crop:
                            # Compute centered crop slices once
                            slices = []
                            for i, max_sz in enumerate(max_shape):
                                if arr.shape[i] > max_sz:
                                    start = (arr.shape[i] - max_sz) // 2
                                    slices.append(slice(start, start + max_sz))
                                else:
                                    slices.append(slice(None))
                            slices = tuple(slices)
                        else:
                            slices = tuple(slice(None, max_sz) for max_sz in max_shape)
                    data[key] = arr[slices]
                    # data[key] = f[key][:128, :512*2, :512*2]
            orig_shape = None
            if image is None:
                image = data[args.key]
            if args.preprocess_volem:
                image = util.convert_white_patches_to_black(image, min_patch_size=100)
        if not args.use_custom_segment:
            seg, pred = segment_mitochondria(
                image,  # model=model,
                model_path=args.model_path,
                scale=scale,
                tiling=tiling,
                return_predictions=True,
                min_size=args.min_size,  # 5000,  # 50000*1,
                seed_distance=args.seed_distance,  # default 6
                ws_block_shape=(128, 256, 256),
                ws_halo=(48, 48, 48),
                boundary_threshold=args.boundary_threshold,
                area_threshold=args.area_threshold,
                preprocess=torch_em.transform.raw.normalize_percentile
            )
        if args.use_custom_segment:
            if args.disk_based_prediction:
                pred_name = os.path.basename(path) + "_axons_pred.zarr"
                if args.tmp_folder is not None:
                    pred_path = os.path.join(args.tmp_folder, pred_name)
                else:
                    pred_path = os.path.join(os.path.dirname(path), pred_name)
                print("using pred path:\n", pred_path)
                spatial_shape = image.shape
                n_out = 1 
                expected_shape = (n_out,) + tuple(spatial_shape)
                # chunk by the *inner* block shape used for writing
                inner_ts = {k: ts[k] - 2 * halo[k] for k in ("z", "y", "x")}
                chunks = (n_out, inner_ts["z"], inner_ts["y"], inner_ts["x"])
                root = zarr.open(pred_path, mode="a")
                pred = root.get("pred", None)
                pred_ready = (
                    pred is not None 
                    and pred.shape == expected_shape 
                    and pred.chunks == chunks
                )
                if not pred_ready:
                    pred = root.create_dataset(
                        "pred",
                        shape=expected_shape,
                        chunks=chunks,
                        dtype="float32",
                        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                        overwrite=True,
                    )
            if not pred_ready:
                pred = get_prediction(
                    input_volume=image,
                    model_path=args.model_path,
                    tiling=tiling,
                    preprocess=torch_em.transform.raw.normalize_percentile,
                    prediction=pred
                    # preprocess=_get_raw_transform
                )
            if not args.only_foreground:
                seg = util.segment_mitos(
                    foreground=pred[0],
                    boundary=pred[1],
                    foreground_threshold=args.foreground_threshold,
                    boundary_threshold=args.boundary_threshold,
                    seed_distance=args.seed_distance,
                    min_size=args.min_size,
                    area_threshold=args.area_threshold,
                    post_iter3d=args.post_iter3d
                )["segmentation"]
            else:
                print("prediciton shape", pred.shape)
                if args.disk_based_prediction:
                    out_name = os.path.basename(path) + "_tmp.zarr"
                    out_path = os.path.join(os.path.dirname(path), out_name)
                    if ".zarr" in output_path:
                        out_path = output_path
                    out_key = args.key if "s" in str(args.key) else "s" + str(args.key)
                    if ".zarr" in args.export_path:
                        out_path = args.export_path
                    seg = util.segment_axons_ooc(
                        pred=pred,
                        out_path=out_path,
                        foreground_threshold=args.foreground_threshold,
                        min_size=args.min_size,
                        out_key=out_key
                    )
                else:
                    #  in memory
                    seg = util.segment_axons(
                        foreground=np.squeeze(pred) if pred.ndim > 3 and pred.shape[0] == 1 else pred,
                        foreground_threshold=args.foreground_threshold,
                        seed_distance=args.seed_distance,
                        min_size=args.min_size,
                        area_threshold=args.area_threshold
                    )
        if args.disk_based_prediction:
            print("Using disk based computations:")
            print(f"Segmentation is stored with key {out_key} at:\n", out_path)
            print("Predictions used are stored at: \n", pred_path)
            return
        with open_file(output_path, "w", ".h5") as f1:
            print("output_path", output_path)
            ndim = seg.ndim
            exp_slicing = tuple(slice(None, None, exp_scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            print("")

            # Save all relevant keys if requested, else save raw
            if args.key is not None and args.all_keys:
                print("keys", keys)
                for key in keys:
                    f1.create_dataset(key, data=(data[key][exp_slicing] if exp_scale != 1 else data[key]), compression="gzip")

            f1.create_dataset("seg", data=(seg[exp_slicing] if exp_scale != 1 else seg), compression="gzip", dtype=seg.dtype)
            if not args.only_foreground:
                f1.create_dataset("pred/foreground", data=(pred[0][exp_slicing] if exp_scale != 1 else pred[0]), compression="gzip", dtype=pred[0].dtype)
                f1.create_dataset("pred/boundary", data=(pred[1][exp_slicing] if exp_scale != 1 else pred[1]), compression="gzip", dtype=pred[0].dtype)
            else:
                f1.create_dataset("pred/foreground", data=(pred[exp_slicing] if exp_scale != 1 else pred), compression="gzip", dtype=pred.dtype)

            # Optionally include additional label datasets
            if args.label_path is not None:
                with open_file(args.label_path, "r") as f2:
                    if args.label_key is not None:
                        f1.create_dataset(f"labels/{args.label_key}", data=f2[args.label_key], compression="gzip", dtype=f2[args.label_key].dtype)
                    else:
                        for key2 in f2.keys():
                            f1.create_dataset(f"labels/{key2}", data=f2[key2], compression="gzip", dtype=f2[key2].dtype)
                print("Saved original segmentation from", args.label_path, "to", output_path)
            print("Saved to", output_path)


if __name__ == "__main__":
    main()
