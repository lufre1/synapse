import argparse
import os
# from glob import glob
# import h5py
# import zarr
# import torch
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import elf.parallel as parallel
import numpy as np
import synapse.io.util as io
# from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d, get_prediction
# from synapse_net.ground_truth.matching import find_additional_objects
# from elf.evaluation.matching import label_overlap, intersection_over_union
# from skimage.segmentation import relabel_sequential
# from skimage.measure import label
# from skimage.transform import resize

VAL_TOMO_PATHS = [
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/KO9_eb13_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/M3_eb4_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/WT21_eb5_model2_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/M7_eb5_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/M7_eb15_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/KO9_eb4_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/M7_eb10_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/M2_eb2_AZ2_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/wichmann_refined/KO9_eb12_model_s4.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/fidi/2025/m13dko/37371_O4_66K_TS_SC_52_rec_2Kb1dawbp_cropF_s2_s2.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/fidi/orig/M13_CTRL_09201_S2_04_DIV31_mtk_04_downscaled_s2.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/fidi/orig/36859_J1_66K_TS_CA3_PS_42_rec_2Kb1dawbp_crop_downscaled_s2.h5',
    '/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all_s4/fidi/orig/37371_O5_66K_TS_SP_34-01_rec_2Kb1dawbp_crop_downscaled_s2.h5'
    ]


def _segment_mitos(
    foreground: np.ndarray,
    boundary: np.ndarray,
    block_shape=(128, 256, 256),
    halo=(32, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    foreground_threshold=0.75,
    min_size=2000,
    area_threshold=200,
    dist=None,
):
    """Return a dict with segmentation, seeds, distance map and h‑map."""
    # ------------------------------------------------------------------
    #  The code you already had – no changes required
    # ------------------------------------------------------------------
    boundaries = boundary
    if dist is None:
        dist = parallel.distance_transform(
            boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape
        )
    hmap = (dist.max() - dist) / (dist.max() + 1e-6)
    # hmap[
    #     np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)
    # ] = (hmap + boundaries).max()
    barrier_mask = np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)
    hmap[barrier_mask] = 1.0

    seeds = np.logical_and(foreground > foreground_threshold, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=True, connectivity=1)
    # seeds = label(seeds, connectivity=2)
    seeds = apply_size_filter(seeds, 250, verbose=True, block_shape=block_shape)

    # mask = (foreground + boundaries) > 0.5
    mask = (foreground + np.where(boundaries < boundary_threshold, boundaries, 0)) > 0.5
    # mask = np.logical_or((foreground > foreground_threshold), (boundary > boundary_threshold))
    # # (boundaries > (1-boundary_threshold)))

    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape, out=seg, verbose=True, halo=halo, mask=mask
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)

    return seg.astype(np.uint8)


def to_string(val):
    s = str(val).replace(".", "")
    return s


def main(args):
    z, y, x = args.tile_shape
    ts = {
        "z": z,
        "y": y,
        "x": x
        }
    halo = {
        "z": int(ts["z"] * 0.25),
        "y": int(ts["y"] * 0.25),
        "x": int(ts["x"] * 0.25)
        }
    tiling = {"tile": ts, "halo": halo}
    os.makedirs(args.export_path, exist_ok=True)
    preprocess = torch_em.transform.raw.normalize_percentile
    mask = None
    verbose = True
    if args.model_path:
        model_path = args.model_path
        model = None
    else:
        raise NotImplementedError
    if args.base_path is None:
        paths = VAL_TOMO_PATHS
    else:
        paths = io.load_file_paths(args.base_path, args.file_extension)
    print("paths", paths)
    for path in tqdm(paths):
        output_path = os.path.join(
            args.export_path,
            os.path.basename(os.path.dirname(args.model_path)).replace(".pt", "") + "_" +
            os.path.basename(path)
                                    )
        output_path = output_path.replace(".zarr", ".h5")
        print("Output path", output_path)
        if os.path.exists(output_path) and not args.force_overwrite:
            print("Skipping... output path exists", output_path)
            continue
        elif os.path.exists(output_path) and args.force_overwrite:
            print("Overwriting... output path exists", output_path)
            os.remove(output_path)
        # load data from file
        data = {}
        with open_file(path, "r") as f:
            for key in [args.key, args.label_key]:
                data[key] = f[key][...]
        # run prediction
        pred = get_prediction(
            data["raw"], model_path=model_path, model=model, tiling=tiling, mask=mask, verbose=verbose,
            preprocess=preprocess
        )
        # run segmentations and save to out path
        for sd in [2, 3, 4, 6, 8,]:
            for bt in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                for ft in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    seg = _segment_mitos(
                        foreground=pred[0],
                        boundary=pred[1],
                        seed_distance=sd,
                        boundary_threshold=bt,
                        foreground_threshold=ft,
                    )
                    with open_file(output_path, "a", ".h5") as f:
                        if args.key not in f:
                            f.create_dataset(args.key, data=data[args.key], compression="gzip",
                                             dtype=data[args.key].dtype)
                        if args.label_key not in f:
                            f.create_dataset(args.label_key, data=data[args.label_key], compression="gzip",
                                             dtype=data[args.label_key].dtype)
                        # add segmentation
                        f.create_dataset(
                            f"seg_sd{to_string(sd)}_bt{to_string(bt)}_ft{to_string(ft)}",
                            data=seg,
                            compression="gzip", dtype=np.uint8,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default=None, help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".h5", help="Path to the root data directory")
    parser.add_argument("--key", "-k",  type=str, default="raw", help="Path to the root data directory")
    parser.add_argument("--label_path", "-lp",  type=str, default=None, help="Path to a specific label file")
    parser.add_argument("--label_key", "-lk",  type=str, default="labels/mitochondria", help="Key to label data within the label file")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/usr/nimlufre/synapse/mitotomo/test_segmentations", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to directory where the model 'best.pt' resides.")
    parser.add_argument("--seed_distance", "-sd", type=int, default=6, help="Seed distance")
    parser.add_argument("--boundary_threshold", "-bt", type=float, default=0.15, help="Boundary threshold")
    parser.add_argument("--foreground_threshold", "-ft", type=float, default=0.75, help="Boundary threshold")
    parser.add_argument("--tile_shape", "-ts", type=int, nargs=3, default=(32, 512, 512), help="Tile shape")
    parser.add_argument("--all_keys", "-ak", default=False, action='store_true', help="If to add all keys from raw file to export file")
    parser.add_argument("--force_overwrite", "-fo", action="store_true", default=False, help="Force overwrite of existing files")
    parser.add_argument("--centered_crop", "-cc", action="store_true", default=False, help="Centered crop")
    parser.add_argument("--downscale_export", "-de", type=int, default=1, help="Downscale export to reduce size")
    
    args = parser.parse_args()
    main(args)