import mrcfile
import imodmodel as imod
import napari
import os
from glob import glob
import numpy as np


def reconstruct_label_mask(imod_data):
    """
    Reconstructs a label mask from IMOD data containing point coordinates.

    Args:
        imod_data (list): A list of lists, where each inner list represents a point
                          with five values (potentially label ID, X, Y, Z, and an unused value).

    Returns:
        numpy.ndarray: A numpy array representing the 3D label mask.

    This function takes IMOD data as input, which is a list of lists where each inner list
    represents a point with five values (label ID, X, Y, Z, and an unused value). It extracts
    the relevant data, finds the unique labels and their corresponding coordinates. It then
    initializes a label mask with zeros and fills it based on the label IDs and coordinates.
    The label mask is returned as a numpy array.

    Note:
        The first column of the imod_data is ignored and the fifth value is assumed to be unused.
        The coordinates are assumed to be integers.
    """

    modified_data = []
    # Extract relevant data (assuming first column is ignored and fifth is unused)
    for row in imod_data:
        label, x, y, z = row[1:]
        label = int(label)
        x = int(x)
        y = int(y)
        z = int(z)
        modified_data.append([label, x, y, z])

    labels, x, y, z = zip(*modified_data)
    unique_labels = np.unique(labels)
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    min_z = min(z)
    max_z = max(z)

    # Initialize label mask with zeros
    mask_shape = (np.ceil(max_z - min_z + 1).astype(int), np.ceil(max_y - min_y + 1).astype(int), np.ceil(max_x - min_x + 1).astype(int))
    label_mask = np.zeros(mask_shape, dtype=int)

    # Fill mask based on label IDs and coordinates
    for label, x_val, y_val, z_val in zip(labels, x, y, z):
        # Adjust coordinates to account for zero-based indexing
        adjusted_x = x_val - min_x
        adjusted_y = y_val - min_y
        adjusted_z = z_val - min_z
        label_mask[adjusted_z, adjusted_y, adjusted_x] = label

    return label_mask


def main(with_mod=True):
    if with_mod:   
        base_path = "/home/freckmann15/data/mitochondria/fidi/20240722_Mito_cristae_segmentation"
        mod_paths = sorted(glob(os.path.join(base_path, "**", "*.mod"), recursive=True))
        mrc_paths = sorted(glob(os.path.join(base_path, "**", "*raw.mrc"), recursive=True))
        mrc_label_paths = sorted(glob(os.path.join(base_path, "**", "*labels.mrc"), recursive=True))
        for mod_path, mrc_path, mrc_label_path in zip(mod_paths, mrc_paths, mrc_label_paths):
            print(f"\nVisualizing \n{mod_path} and \n{mrc_path} and \n{mrc_label_path} \n")
            with mrcfile.open(mrc_path) as raw:
                with mrcfile.open(mrc_label_path) as mrc_labels:
                    labels = imod.read(mod_path)
                    v = napari.Viewer()
                    v.add_image(raw.data)
                    v.add_labels(reconstruct_label_mask(labels.to_numpy("float")))
                    v.add_labels(mrc_labels.data)
                    napari.run()
    else:
        base_path = "/home/freckmann15/data/mitochondria/fidi/20240722_Mito_cristae_segmentation"
        mrc_paths = sorted(glob(os.path.join(base_path, "**", "*raw.mrc"), recursive=True))
        mrc_label_paths = sorted(glob(os.path.join(base_path, "**", "*labels.mrc"), recursive=True))
        for mrc_path, mrc_label_path in zip(mrc_paths, mrc_label_paths):
            print(f"\nVisualizing \n{mrc_path} and \n{mrc_label_path} \n")
            with mrcfile.open(mrc_path) as raw:
                with mrcfile.open(mrc_label_path) as mrc_labels:
                    v = napari.Viewer()
                    v.add_image(raw.data)
                    v.add_labels(mrc_labels.data)
                    napari.run()


if __name__ == "__main__":
    main(False)
