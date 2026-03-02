import h5py
import numpy as np
import tifffile
# import mrcfile
from glob import glob
import argparse
import os
from synapse.empanada_util import get_empanada_config
from skimage.transform import rescale, resize

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
    
    # elif ext in {"mrc", "rec"}:
    #     if not isinstance(data, np.ndarray):
    #         raise ValueError("For .mrc and .rec formats, data must be a NumPy array.")
        # with mrcfile.new(export_path, overwrite=True) as mrc:
        #     mrc.set_data(data.astype(data.dtype))

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


def adjust_size(input_volume, scale=None, is_segmentation=False, orig_shape=None):
    """
    Rescale or resize a 2D/3D volume, using interpolation appropriate for images vs. label maps.

    This function has two modes:

    1) Rescaling (when ``orig_shape is None``):
       - Uses ``skimage.transform.rescale`` with the provided ``scale``.

    2) Resizing to a target shape (when ``orig_shape is not None``):
       - Uses ``skimage.transform.resize`` to match ``orig_shape``.

    For segmentation/label volumes (``is_segmentation=True``), nearest-neighbor interpolation
    is used (``order=0`` and ``anti_aliasing=False``) to avoid creating non-integer labels.
    For intensity images (``is_segmentation=False``), default interpolation is used.

    Parameters
    ----------
    input_volume : np.ndarray
        Input image/volume (2D or 3D). The output is cast back to ``input_volume.dtype``.
    scale : float or sequence of float, optional
        Scale factor(s) passed to ``rescale``. Required when ``orig_shape`` is None.
        Examples: ``0.5`` to downsample by 2, or ``(1, 0.5, 0.5)`` for anisotropic scaling.
    is_segmentation : bool, default=False
        If True, treat ``input_volume`` as a label map and use nearest-neighbor interpolation.
    orig_shape : tuple of int, optional
        Target output shape passed to ``resize``. If provided, ``scale`` is ignored.

    Returns
    -------
    np.ndarray
        Rescaled/resized volume with the same dtype as the input.

    Notes
    -----
    - ``preserve_range=True`` is used to avoid normalization to [0, 1] by scikit-image.
    - For segmentation resizing, nearest-neighbor interpolation preserves label identities.
    """
    if orig_shape is None:
        if is_segmentation:
            input_volume = rescale(
                input_volume, scale, preserve_range=True, order=0, anti_aliasing=False,
            ).astype(input_volume.dtype)
        else:
            input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
    else:
        if is_segmentation:
            input_volume = resize(input_volume, orig_shape, preserve_range=True, order=0, anti_aliasing=False).astype(input_volume.dtype)
        else:
            input_volume = resize(input_volume, orig_shape, preserve_range=True).astype(input_volume.dtype)
    return input_volume


def segment_mitochondria(path, visualize=False, scale=1, z_slice=None, args=None) -> dict:
    from empanada_napari.inference import Engine3d, Engine2d
    scaler = args.scaler
    # path = "/home/freckmann15/.cache/synapse-net/sample_data/mito_small.mrc"
    # load data
    if ".mrc" in path:
        return
        # with mrcfile.open(path) as mrc:
        #     data = mrc.data
    elif ".h5" in path:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            # print(keys)
            if "raw" in keys:
                ndim = f["raw"].ndim
                # print("data.shape", f["raw"].shape)
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

                # Apply downsampling while preserving batch/channel dimensions
                data = f["raw"][slicing] if scale > 1 else f["raw"][:]

                # data = data[::, ::6, ::6]
                # print("data.shape", data.shape)
                # data = f["raw"][:]
            else:
                print("Could not find raw data in file", path)
                return
            # mito_key = [key for key in data.keys() if "labels/mitochondria" in key] or None
            # if "labels/mitochondria" in keys:
            #     mito_key = "labels/mitochondria"
            # else:
            #     mito_key = None
            mito_key = "labels/mitochondria"
            if mito_key is not None:
                ndim = f[mito_key].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                mitos = f[mito_key][slicing] if scale > 1 else f[mito_key][:]
            else:
                mitos = None
            if data is not None:
                if scaler is not None:
                    original_shape = data.shape
                    data = adjust_size(data, scaler, is_segmentation=False)

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

    if scaler is not None:
        original_shape = data.shape
        stack = adjust_size(stack, scaler, is_segmentation=False, orig_shape=original_shape)

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
            # adjust mito shape
            mitos = adjust_size(
                mitos.astype(np.uint8),
                scale=1,
                is_segmentation=True,
                orig_shape=data.shape
                )
            return {
                "raw": data,
                "labels/mitochondria": mitos,
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
            export_path = os.path.join(args.output_path, f"{filename}_empanada.h5")
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            print("Created output folder:", args.output_path)
        export_data(export_path, data)


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--path", "-p", type=str, default="/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_s4")
    argsparse.add_argument("--output_path", "-o", type=str, default="/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/out_empanada_vs6nm")
    argsparse.add_argument("--visualize", "-v", action="store_true", default=False)
    argsparse.add_argument("--z_slice", "-z", type=int, default=None, help="Segment 2d slice: z")
    argsparse.add_argument("--z_range", "-r", type=int, nargs=2, default=None, help="Use stacked 2d slices for 3d")
    argsparse.add_argument("--scale", "-s", type=int, default=1, help="Downsample data while reading")
    argsparse.add_argument("--scaler", "-sr", type=int, default=None, help="Use empanada scaler during inference")
    argsparse.add_argument("--tile_size", "-t", type=int, default=0)
    argsparse.add_argument("--downsample", "-d", type=int, default=1)
    argsparse.add_argument("--lucchi", "-l", action="store_true", default=False)
    args = argsparse.parse_args()
    main(args)
