import h5py
import numpy as np
import tifffile
import mrcfile
from empanada_napari.inference import Engine3d, Engine2d
from glob import glob
import argparse
import os
from synapse.empanada_util import get_empanada_config
# import synapse.io.util as util
# from elf.io import open_file


def export_data(export_path: str, data):
    """Export data to the specified path, determining format from the file extension.
    
    Args:
        data (np.ndarray | dict): The data to save. For HDF5/Zarr, a dict of named datasets is required.
        export_path (str): The file path where the data should be saved.
    
    Raises:
        ValueError: If the file format is unsupported or if data format does not match the expected type.
    """
    ext = export_path.lower().split(".")[-1]

    if ext == "tif":
        if isinstance(data, dict):
            data = next(iter(data.values()))
        if not isinstance(data, np.ndarray):
            raise ValueError("For .tif format, data must be a NumPy array.")
        tifffile.imwrite(export_path, data, dtype=data.dtype, compression="zlib")
        # iio.imwrite(export_path, data, compression="zlib")
    
    elif ext in {"mrc", "rec"}:
        if not isinstance(data, np.ndarray):
            raise ValueError("For .mrc and .rec formats, data must be a NumPy array.")
        with mrcfile.new(export_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(data.dtype))

    elif ext in {"h5", "hdf5"}:
        if not isinstance(data, dict):
            raise ValueError("For .h5 and .hdf5 formats, data must be a dictionary with dataset names as keys.")
        with h5py.File(export_path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value.astype(value.dtype))
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Data successfully exported to {export_path}")


def _get_file_paths():
    input_files = [
        "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5",
        "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/WT21_eb3_model2.h5",
        "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M10_eb9_model.h5",
        "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/KO9_eb4_model.h5",
        "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M7_eb11_model.h5",
        "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_PS_25_rec_2Kb1dawbp_crop_downscaled.h5"
    ]
    return input_files


def segment_mitochondria(path, visualize=False, scale=1) -> dict:
    # path = "/home/freckmann15/.cache/synapse-net/sample_data/mito_small.mrc"

    config = get_empanada_config()
    # engine = Engine3d(model_config=config, save_panoptic=True)
    engine = Engine2d(model_config=config, tile_size=512)

    # load data
    if ".mrc" in path:
        with mrcfile.open(path) as mrc:
            data = mrc.data
    elif ".h5" in path:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(keys)
            if "raw" in keys:
                ndim = f["raw"].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

                # Apply downsampling while preserving batch/channel dimensions
                data = f["raw"][slicing] if scale > 1 else f["raw"][:]
                # data = f["raw"][:]
            else:
                print("Could not find raw data in file", path)
                return
            # mito_key = [key for key in data.keys() if "labels/mitochondria" in key] or None
            mito_key = "labels/mitochondria"
            if mito_key is not None:
                ndim = f[mito_key].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                mitos = f[mito_key][slicing] if scale > 1 else f[mito_key][:]
            else:
                mitos = None
    # print(mito_key)
    volume = data.astype(np.uint8)

    # stack, trackers = engine.infer_on_axis(volume=volume, axis_name="xy")  # {'xy': 0, 'xz': 1, 'yz': 2}
    stack = engine.infer(volume)

    if visualize:
        import napari
        v = napari.Viewer()
        v.add_image(data)
        v.add_labels(stack, name="segmentation")
        if mitos:
            v.add_labels(mitos, name=mito_key)
        napari.run()
    else:
        return {
            "raw": data,
            "labels/mitochondria": mitos,
            "seg": stack
        }


def main(args):
    if args.path is None:
        raw_paths = _get_file_paths()
    else:
        root_path = args.path
        raw_paths = sorted(glob(os.path.join(root_path, "**", "*.h5"), recursive=True), reverse=True)
    print(f"Found {len(raw_paths)} files:")
    for path in raw_paths:
        data = segment_mitochondria(path)
        filename = os.path.basename(path).split(".")[0]
        export_path = os.path.join(args.output_path, f"{filename}.h5")
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print("Created output folder:", args.output_path)
        export_data(export_path, data)


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--path", "-p", type=str, default=None)  # /home/freckmann15/data/mitochondria/eval_mitov3/script_luca_with_pred_and_tifs_exported
    argsparse.add_argument("--output_path", "-o", type=str, default="out_empanada")
    args = argsparse.parse_args()
    main(args)