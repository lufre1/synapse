
import os
import h5py
import numpy as np
import torch
import torch_em
from tqdm import tqdm
from synapse_net.inference.cristae import segment_cristae
import argparse
import synapse.util as util
from synapse_net.inference.util import parse_tiling


def segment(args, paths):
    n_workers = 8 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} with {n_workers} workers.")
    print(args)
    tiling = parse_tiling(args.tile_size, args.halo)

    for path in tqdm(paths, desc="Segmenting Cristae"):
        with h5py.File(path, "r") as f:
            volume = util.standardize_channel(f["raw_mitos_combined"][:])
        kwargs = {
            "extra_segmentation": volume[1]
        }
        seg, pred = segment_cristae(volume[0], args.checkpoint_path, tiling, return_predictions=True,
                                    krwargs=kwargs)
        filename = os.path.basename(path).replace(".h5", "cristae_seg.h5")
        out_path = os.path.join(args.output_path, filename)
        os.makedirs(args.output_path, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="raw", data=volume[0])
            f.create_dataset(name="labels/mitos", data=volume[1], compression="gzip", dtype=np.dtype("uint8"))
            f.create_dataset(name="pred", data=pred)
            f.create_dataset(name="seg", data=seg, compression="gzip", dtype=np.dtype("uint8"))
        print(f"Saved segmentation to {out_path}")
    


def _get_data_paths():
    # this is train split for cristae
    data_paths = [
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_eb7_AZ1_model2_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_syn7_model2_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_syn2_model2_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-KO_P10/M1_eb1_model_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT13_syn6_model2_combined.h5',
        '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-KO_P22/M7_eb12_model_combined.h5'
    ]
    return data_paths


def main():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--tile_size", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--halo", type=int, nargs=3, default=(8, 64, 64), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--checkpoint_path", type=str, required=True, default=None, help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="default-cristae-segmentations", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="Path to the output directory")
    args = parser.parse_args()

    paths = _get_data_paths()
    segment(args, paths)


if __name__ == "__main__":
    main()