import h5py
import numpy as np
import tifffile
import mrcfile
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


def segment_mitochondria(path, visualize=False, scale=1, z_slice=None, args=None) -> dict:
    from empanada_napari.inference import Engine3d, Engine2d
    # path = "/home/freckmann15/.cache/synapse-net/sample_data/mito_small.mrc"

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
                print("data.shape", data.shape)
                data = data[::6, ::2, ::2]
                print("data.shape", data.shape)
                # data = f["raw"][:]
            else:
                print("Could not find raw data in file", path)
                return
            # mito_key = [key for key in data.keys() if "labels/mitochondria" in key] or None
            if "labels/mitochondria" in keys:
                mito_key = "labels/mitochondria"
            else:
                mito_key = None
            if mito_key is not None:
                ndim = f[mito_key].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                mitos = f[mito_key][slicing] if scale > 1 else f[mito_key][:]
            else:
                mitos = None
    # print(mito_key)
    volume = data.astype(np.uint8)

    config = get_empanada_config()

    if z_slice is not None:
        engine = Engine2d(model_config=config, tile_size=args.tile_size, inference_scale=args.downsample)
        volume = volume[z_slice, :, :]
        mitos = mitos[z_slice, :, :]
        stack = engine.infer(volume)
        data = data[z_slice, :, :]
    elif args.z_range is not None:
        start, end = args.z_range
        engine = Engine2d(model_config=config, tile_size=args.tile_size, inference_scale=args.downsample)
        mitos = mitos[start:end, :, :]
        data = data[start:end, :, :]
        stack = np.zeros(shape=data.shape, dtype=np.uint8)
        for z in range(volume[start:end, :, :].shape[0]):
            stack[z] = engine.infer(volume[z])
    else:
        engine = Engine3d(model_config=config, save_panoptic=True, inference_scale=args.downsample)
        stack, trackers = engine.infer_on_axis(volume=volume, axis_name="xy")  # {'xy': 0, 'xz': 1, 'yz': 2}

    if visualize:
        import napari
        v = napari.Viewer()
        v.add_image(data)
        v.add_labels(stack, name="segmentation")
        if mitos:
            v.add_labels(mitos, name=mito_key)
        napari.run()
    else:
        if mitos is None:
            return {
                "raw": data,
                "seg": stack.astype(np.uint8)
            }
        else:
            return {
                "raw": data,
                "labels/mitochondria": mitos.astype(np.uint8),
                "seg": stack.astype(np.uint8)
            }


def main(args):
    if args.path is None:
        raw_paths = _get_file_paths()
    else:
        root_path = args.path
        if os.path.isdir(root_path):
            raw_paths = sorted(glob(os.path.join(root_path, "**", "*.h5"), recursive=True), reverse=True)
        else:
            raw_paths = [root_path]
    if args.lucchi:
        raw_paths = [
            "/home/freckmann15/data/lucchi/lucchi_test.h5"
            ]
    print(f"Found {len(raw_paths)} files:")
    for path in raw_paths:
        data = segment_mitochondria(path, args.visualize,
                                    z_slice=args.z_slice, scale=args.scale,
                                    args=args)
        filename = os.path.basename(path).split(".")[0]
        if args.tile_size > 0 or args.z_slice:  # 2d
            export_path = os.path.join(args.output_path, f"{filename}_ts{args.tile_size}_z{args.z_slice}_ds{args.downsample}.h5")
        elif args.z_range is not None:  # 3d with z_range (stacked 2d)
            export_path = os.path.join(args.output_path, f"{filename}_z{args.z_range[0]}-{args.z_range[1]}_ds{args.downsample}.h5")
        else:  # 3d with full volume (panoptic from empanada)
            export_path = os.path.join(args.output_path, f"{filename}_ds{args.downsample}.h5")
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print("Created output folder:", args.output_path)
        export_data(export_path, data)


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--path", "-p", type=str, default=None)  # /home/freckmann15/data/mitochondria/eval_mitov3/script_luca_with_pred_and_tifs_exported
    argsparse.add_argument("--output_path", "-o", type=str, default="out_empanada")
    argsparse.add_argument("--visualize", "-v", action="store_true", default=False)
    argsparse.add_argument("--z_slice", "-z", type=int, default=None)
    argsparse.add_argument("--z_range", "-r", type=int, nargs=2, default=None)
    argsparse.add_argument("--scale", "-s", type=int, default=1)
    argsparse.add_argument("--tile_size", "-t", type=int, default=0)
    argsparse.add_argument("--downsample", "-d", type=int, default=1)
    argsparse.add_argument("--lucchi", "-l", action="store_true", default=False)
    args = argsparse.parse_args()
    main(args)
