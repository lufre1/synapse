import argparse
import mrcfile
import os
import h5py
import numpy as np
import tifffile
from tqdm import tqdm
from glob import glob
from skimage.transform import resize


def resize_to_shape(data, shape, is_segmentation=False):
    if is_segmentation:
        return resize(
            data, shape, preserve_range=True, order=0,
            anti_aliasing=False
            ).astype(np.uint16)
    else:
        return resize(
            data, shape, preserve_range=True, order=1,
            anti_aliasing=False
            ).astype(np.uint8)


def copy_volume_in_chunks(dst_ds, src, z_chunk=16):
    """Copy 3D array-like src -> HDF5 dataset dst_ds in z-chunks."""
    z = src.shape[0]
    for z0 in range(0, z, z_chunk):
        z1 = min(z0 + z_chunk, z)
        dst_ds[z0:z1] = src[z0:z1]


def export_one(rp, mp, cp, output_file, z_chunk=16, compression="gzip", compression_opts=4):
    with mrcfile.open(rp, permissive=True) as mrc:
        raw_src = mrc.data  # memmap-like for many MRCs
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
        shape = raw_src.shape
        raw_dtype = raw_src.dtype

        # open TIFFs as memmaps (important: use out='memmap')
        mito_src = tifffile.imread(mp, out="memmap")
        cristae_src = tifffile.imread(cp, out="memmap")

        if mito_src.shape != shape or cristae_src.shape != shape:
            raise ValueError(f"Shape mismatch: raw {shape}, mito {mito_src.shape}, cristae {cristae_src.shape}")

        # Flip raw data along axis 1
        raw_src = np.flip(raw_src, axis=1)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            f.attrs["axes"] = np.array(["z", "y", "x"], dtype=h5py.string_dtype(encoding="utf-8"))
            f.attrs["voxel_unit"] = "nm"
            f.attrs["voxel_size"] = np.asarray(voxel_size, dtype=np.float32)

            chunk = (min(z_chunk, shape[0]), shape[1], shape[2])  # chunk along z
            ds_raw = f.create_dataset("raw", shape=shape, dtype=raw_dtype,
                                      chunks=chunk, compression=compression, compression_opts=compression_opts)
            ds_mito = f.create_dataset("labels/mitochondria", shape=shape, dtype=mito_src.dtype,
                                       chunks=chunk, compression=compression, compression_opts=compression_opts)
            ds_cri = f.create_dataset("labels/cristae", shape=shape, dtype=cristae_src.dtype,
                                      chunks=chunk, compression=compression, compression_opts=compression_opts)

            copy_volume_in_chunks(ds_raw, raw_src, z_chunk=z_chunk)
            copy_volume_in_chunks(ds_mito, mito_src, z_chunk=z_chunk)
            copy_volume_in_chunks(ds_cri, cristae_src, z_chunk=z_chunk)


def crop_data(raw, labels_dict):
    """
    Crop the raw data and all label datasets in labels_dict to a subset containing only slices with label data.

    Parameters
    ----------
    raw : numpy.ndarray
        The raw data to crop.
    labels_dict : dict
        A dictionary with label names as keys and the corresponding label datasets as values.

    Returns
    -------
    raw_cropped : numpy.ndarray
        The cropped raw data.
    cropped_labels_dict : dict
        A dictionary with the same keys as labels_dict, but with the cropped label datasets as values.

    """
    combined_labels = np.zeros_like(next(iter(labels_dict.values())))
    for labels in labels_dict.values():
        combined_labels |= labels

    # Identify slices along the z-dimension where there is label data
    non_zero_slices = np.any(combined_labels, axis=(1, 2))

    # Crop raw data
    raw_cropped = raw[non_zero_slices, :, :]

    # Crop each label dataset in labels_dict
    cropped_labels_dict = {
        label_name: labels[non_zero_slices, :, :]
        for label_name, labels in labels_dict.items()
    }

    return raw_cropped, cropped_labels_dict


def main():
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/20260320_for_Luca", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/home/freckmann15/data/mitochondria/cooper/20260320_for_Luca_export", help="Path to the root data directory")
    parser.add_argument("--visualize", "-v", default=False, action='store_true', help="If to visualize or not")
    parser.add_argument("--print_labels", "-pl", default=False, action='store_true', help="If to print labels from mod file or not")
    parser.add_argument("--force_overwrite", "-f", default=False, action='store_true', help="If to over-write already present segmentation results.")
    args = parser.parse_args()

    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mrc"), recursive=True))
    tif_paths = sorted(glob(os.path.join(args.base_path, "**", "*.tif"), recursive=True))
    mito_paths = [path for path in tif_paths if "prediction_mito_new" in path]
    cristae_paths = [path for path in tif_paths if "prediction_cristae_new" in path]
    for rp, mp, cp in tqdm(zip(mrc_paths, mito_paths, cristae_paths), total=len(mrc_paths)):
        print(f"raw path, mito path, cristae path:\n{rp}\n{mp}\n{cp}\n")
        output_file = os.path.join(args.export_path, os.path.basename(rp).replace(".mrc", ".h5"))
        if os.path.exists(output_file):
            print("File already exists")
            if not args.force_overwrite:
                continue
            else:
                os.remove(output_file)
        export_one(rp, mp, cp, output_file)
        # with mrcfile.open(rp) as mrc:
        #     raw = mrc.data
        #     voxel_size = mrc.voxel_size
        # with h5py.File(output_file, "w") as f:
        #     f.create_dataset("raw", data=raw, dtype=raw.dtype, compression="gzip")
        #     f.attrs.create(name="voxel_size", data=voxel_size, shape=voxel_size.shape)
        # del raw
        # mito = tifffile.imread(mp)
        # cristae = tifffile.imread(cp)
        # with h5py.File(output_file, "a") as f:
        #     f.create_dataset("label_mito", data=mito, dtype=mito.dtype, compression="gzip")
        #     f.create_dataset("label_cristae", data=cristae, dtype=cristae.dtype, compression="gzip")
        # del mito
        # del cristae
        print("Exported to", output_file)


if __name__ == "__main__":
    main()