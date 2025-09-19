import os
import numpy as np
import h5py
from imageio import imread
import os
import numpy as np
import h5py
from imageio import imread


def sorted_files(directory, extensions):
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(extensions)])


def crop_slice(slice_img, crop_shape, centered=False):
    cropped_slices = []
    for dim, size in enumerate(crop_shape):
        slc = slice_img.shape[dim]
        if size >= slc:
            cropped_slices.append(slice(None))
        else:
            if centered:
                start = (slc - size) // 2
            else:
                start = 0
            cropped_slices.append(slice(start, start + size))
    return slice_img[tuple(cropped_slices)]


def process_and_save_stack(image_folder, extensions, h5_dataset, crop_shape, start_idx=0, centered=False):
    files = sorted_files(image_folder, extensions)
    for i, fname in enumerate(files):
        if start_idx + i >= h5_dataset.shape[0]:  # <-- Add this bounds check
            break
        img = imread(os.path.join(image_folder, fname))
        cropped = crop_slice(img, crop_shape[1:], centered=centered)  # crop y,x dims only
        h5_dataset[start_idx + i, ...] = cropped


if __name__ == "__main__":
    raw_folder = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/im_pad"
    label_folder = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/mito-train-v2"
    output_h5 = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/mitoem_volume_cropped.h5"
    crop_shape = (200, 1024, 1024)  # (Z,Y,X)

    raw_files = sorted_files(raw_folder, ('.png',))
    label_files = sorted_files(label_folder, ('.tif', '.tiff'))

    # Determine how many slices to write (crop or limit by available slices)
    raw_slices = min(len(raw_files), crop_shape[0])
    label_slices = min(len(label_files), crop_shape[0])

    with h5py.File(output_h5, 'w') as f:
        raw_dset = f.create_dataset('raw', shape=(raw_slices, crop_shape[1], crop_shape[2]),
                                dtype=np.uint8, compression='gzip')
        label_dset = f.create_dataset('labels/mitochondria', shape=(label_slices, crop_shape[1], crop_shape[2]),
                                    dtype=np.uint16, compression='gzip')

        process_and_save_stack(raw_folder, ('.png',), raw_dset, crop_shape, centered=False)
        process_and_save_stack(label_folder, ('.tif', '.tiff'), label_dset, crop_shape, centered=False)

    print(f"Saved cropped volume to {output_h5}")



# def load_tif_stack(directory):
#     """
#     Load a stack of TIFF (.tif or .tiff) images from a directory as a 3D numpy array (z, y, x).
#     Assumes filenames are sortable and represent slice order.
    
#     Parameters:
#         directory (str): Path to folder containing TIFF files.
    
#     Returns:
#         numpy.ndarray: Stacked 3D volume array.
#     """
#     tif_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.tif', '.tiff'))])
#     if not tif_files:
#         raise RuntimeError(f"No TIFF files found in {directory}")
    
#     first_img = imread(os.path.join(directory, tif_files[0]))
#     volume = np.zeros((len(tif_files),) + first_img.shape, dtype=first_img.dtype)
#     for i, fname in enumerate(tif_files):
#         volume[i] = imread(os.path.join(directory, fname))
#     return volume


# def load_png_stack(directory):
#     png_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.png')])
#     if not png_files:
#         raise RuntimeError(f"No PNG files found in {directory}")
#     first_img = imread(os.path.join(directory, png_files[0]))
#     volume = np.zeros((len(png_files),) + first_img.shape, dtype=first_img.dtype)
#     for i, fname in enumerate(png_files):
#         volume[i] = imread(os.path.join(directory, fname))
#     return volume


# def crop_volume(volume, crop_shape, centered=False):
#     """
#     Crop the volume to provided shape (z, y, x).
    
#     Parameters:
#         volume (np.ndarray): Input 3D volume.
#         crop_shape (tuple): Desired output shape (z, y, x).
#         center (bool): If True (default), crops centered. If False, crops from start (upper-left corner).
    
#     Returns:
#         Cropped volume as np.ndarray.
#     """
#     cropped_slices = []
#     for i, size in enumerate(crop_shape):
#         vol_size = volume.shape[i]
#         if size >= vol_size:
#             cropped_slices.append(slice(None))
#         else:
#             if centered:
#                 start = (vol_size - size) // 2
#             else:
#                 start = 0
#             cropped_slices.append(slice(start, start + size))
#     return volume[tuple(cropped_slices)]


# def save_to_hdf5(volume_dict, out_path):
#     with h5py.File(out_path, 'w') as f:
#         for key, volume in volume_dict.items():
#             f.create_dataset(key, data=volume, compression="gzip")


# if __name__ == "__main__":
#     # CHANGE THESE PATHS as needed
#     raw_folder = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/im_pad"
#     label_folder = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/mito-train-v2"
#     output_h5 = "/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/MitoEM/mitoem_volume_cropped.h5"

#     crop_shape = (200, 1024, 1024)  # desired crop shape (Z,Y,X)

#     raw_vol = load_png_stack(raw_folder)
#     label_vol = load_tif_stack(label_folder)

#     raw_cropped = crop_volume(raw_vol, crop_shape)
#     label_cropped = crop_volume(label_vol, crop_shape)

#     print("Cropped raw shape:", raw_cropped.shape)
#     print("Cropped label shape:", label_cropped.shape)

#     save_to_hdf5({'raw': raw_cropped, 'labels/mitochondria': label_cropped}, output_h5)
#     print(f"Saved cropped volumes to {output_h5} as datasets 'raw' and 'labels'")
