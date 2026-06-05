import argparse
import os
from glob import glob
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
import synapse.label_utils as lutil
import synapse.prediction as pred_util
from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.util import get_prediction
# from synapse_net.ground_truth.matching import find_additional_objects
from skimage.measure import label
from skimage.transform import resize
import argparse
import yaml

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
    p.add_argument("--bg_penalty", "-bp", type=float, default=2.0,
                   help="Height-map penalty for barrier voxels in ooc watershed. "
                        "Lower values (e.g. 1.2) reduce fragmentation at thin necks.")
    p.add_argument("--n_threads", "-nt", type=int, default=8,
                   help="Number of threads for parallel OOC operations. "
                        "Set to match SLURM -c allocation to avoid OOM from using all node CPUs.")
    p.add_argument("--keep_intermediates", "-ki", action="store_true",
                   help="Keep intermediate OOC datasets (dist, seeds) in the output zarr. "
                        "By default they are deleted after segmentation.")
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

def main(visualize=False):

    args = parse_args() 
    exp_scale = args.downscale_export
    add_missing_mitos = args.add_missing_mitos
    print(args.base_path)
    print("\nUsing model", args.model_path)
    # tile_shape
    tiling = pred_util.make_tiling(args.tile_shape)
    ts = tiling["tile"]
    halo = tiling["halo"]
    h5_paths = io.load_file_paths(args.base_path, args.file_extension)

    print("len(h5_paths)", len(h5_paths))
    # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = None
    bt_string = str(args.boundary_threshold).replace(".", "")
    ft_string = str(args.foreground_threshold).replace(".", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using best model from", args.model_path, "with device", device)

    for path in tqdm(h5_paths):

        print("\nopening file", path)
        if ".zarr" not in args.export_path:
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
        else:
            output_path = args.export_path
        if path.endswith(".h5"):
            keys = io.get_all_keys_from_h5(path)
        else:
            keys = io.get_all_dataset_keys(path)
        data = {}

        with open_file(path, "r") as f:
            centered_crop = args.centered_crop
            if args.key is not None and not args.all_keys:
                image = f[args.key]
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
                image = util.convert_white_patches_to_black(image)
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
            pred = None
            pred_ready = False
            if args.disk_based_prediction:
                n_out = 2  # foreground + boundary
                pred_name = os.path.basename(path) + "_pred.zarr"
                pred_path = os.path.join(os.path.dirname(path), pred_name)

                print("image shape", image.shape)
                inner_ts = pred_util.inner_tile_shape(tiling)
                # On force_overwrite, drop any cached prediction so the helper
                # recreates (and clears) the on-disk dataset before recomputing.
                if args.force_overwrite:
                    _root = zarr.open(pred_path, mode="a")
                    if "pred" in _root:
                        del _root["pred"]
                        print("Force overwrite: discarding cached prediction, will recompute.")
                pred, pred_ready = pred_util.open_disk_prediction(
                    pred_path, image.shape, inner_ts, n_out=n_out,
                    use_done_marker=False, verbose=False,
                )
                print("Prediction needs to be recomputed:", not pred_ready)
                print("Disk based prediction:", args.disk_based_prediction)
                print("prediction path", pred_path)
                print("prediction shape", (n_out,) + tuple(image.shape))
                print("Prediction already computed", pred_ready)
                
            if not pred_ready:
                pred = get_prediction(
                    input_volume=image,
                    model_path=args.model_path,
                    tiling=tiling,
                    preprocess=torch_em.transform.raw.normalize_percentile,
                    prediction=pred,
                )
            if args.disk_based_prediction:
                occ_filename = os.path.basename(path) + "_wrapper_tmp.zarr"
                occ_path = os.path.join(os.path.dirname(path), occ_filename)
                if ".zarr" in output_path:
                    occ_path = output_path
                seg = util.segment_mitos_ooc_wrapped(
                    pred=pred,
                    foreground_threshold=args.foreground_threshold,
                    boundary_threshold=args.boundary_threshold,
                    seed_distance=args.seed_distance,
                    min_size=args.min_size,
                    area_threshold=args.area_threshold,
                    out_dir=occ_path,
                    reuse_computed=False,
                    bg_penalty=args.bg_penalty,
                    n_threads=args.n_threads,
                )["segmentation"]
                if not args.keep_intermediates:
                    pred_util.delete_zarr_intermediates(occ_path)
            else:
                occ_path = None
                seg = util.segment_mitos(
                    foreground=pred[0],
                    boundary=pred[1],
                    foreground_threshold=args.foreground_threshold,
                    boundary_threshold=args.boundary_threshold,
                    seed_distance=args.seed_distance,
                    min_size=args.min_size,
                    area_threshold=args.area_threshold,
                    post_iter3d=args.post_iter3d,
                )["segmentation"]
        # skip export if using desk based predictions / segmentations
        if args.disk_based_prediction:
            print("Using disk based computations:")
            print("Segmentation is stored at:\n", occ_path)
            print("Predictions used are stored at: \n", pred_path)
            return
        else:        
            with open_file(output_path, "w", ".h5") as f1:
                print("output_path", output_path)
                ndim = seg.ndim
                exp_slicing = tuple(slice(None, None, exp_scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                print("")

                # Save all relevant keys if requested, else save raw
                if args.key is not None and args.all_keys:
                    print("keys", keys)
                    for key in keys:
                        if "mito" in key and add_missing_mitos:
                            added = label(data[key] + lutil.find_additional_objects(data[key], seg, matching_threshold=0.1))
                            f1.create_dataset(key, data=(added[exp_slicing] if exp_scale != 1 else added), compression="gzip")
                        else:
                            f1.create_dataset(key, data=(data[key][exp_slicing] if exp_scale != 1 else data[key]), compression="gzip")

                    f1.create_dataset("pred/foreground", data=(pred[0][exp_slicing] if exp_scale != 1 else pred[0]), compression="gzip", dtype=pred[0].dtype)
                    f1.create_dataset("pred/boundary", data=(pred[1][exp_slicing] if exp_scale != 1 else pred[1]), compression="gzip", dtype=pred[0].dtype)
                if args.disk_based_prediction:
                    util.export_ooc_to_h5(seg, f1, "seg", exp_scale=exp_scale, chunk_shape=(128, 256, 256))
                else:
                    f1.create_dataset("seg", data=(seg[exp_slicing] if exp_scale != 1 else seg), compression="gzip", dtype=seg.dtype)

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
