import h5py
import mrcfile
import os
from glob import glob
import numpy as np
import napari
import argparse
from magicgui import magicgui
from skimage.measure import label
# from scipy import ndimage
from napari.utils.notifications import show_info
from tqdm import tqdm
from skimage.measure import regionprops

SAVE_DIR = "/home/freckmann15/data/mitochondria/corrected_mito_h5"
BASE_PATH = "/home/freckmann15/data/mitochondria/20240722_Mito_cristae_segmentation/"


def label_image_in_chunks(image, chunk_size):
    """Labels an image in chunks to reduce memory usage.

    Args:
        image: The input image as a NumPy array.
        chunk_size: The size of the chunks to process.

    Returns:
        A labeled image as a NumPy array.
    """

    shape = image.shape
    labels = np.zeros_like(image, dtype=np.int32)

    for z in range(0, shape[2], chunk_size):
        for y in range(0, shape[1], chunk_size):
            for x in range(0, shape[0], chunk_size):
                chunk = image[x:x+chunk_size, y:y+chunk_size, z:z+chunk_size]
                chunk_labels = label(chunk)
                # Adjust labels for overlap if necessary
                labels[x:x+chunk_size, y:y+chunk_size, z:z+chunk_size] = chunk_labels

    return labels


def extend_corner_regions(labels):
    """Extends regions that touch two borders of the image to fill the corner.

    Args:
        labels: A 3D NumPy array containing the labels.

    Returns:
        A 3D NumPy array with extended labels.
    """

    label_image = label_image_in_chunks(labels, 512)
    # label_image = label(labels)        # if iteration == 1:
    #         continue
    regions = regionprops(label_image)

    for prop in regions:
        minr, minc, minz, maxr, maxc, maxz = prop.bbox
        shape = labels.shape

        # Check for corners
        if minr == 0 and minc == 0:
            labels[minr, minc, minz:maxz+1] = prop.label
        elif minr == 0 and maxc == shape[1] - 1:
            labels[minr, maxc, minz:maxz+1] = prop.label
        elif maxr == shape[0] - 1 and minc == 0:
            labels[maxr, minc, minz:maxz+1] = prop.label
        elif maxr == shape[0] - 1 and maxc == shape[1] - 1:
            labels[maxr, maxc, minz:maxz+1] = prop.label
        # Similar checks for other corners in the Z dimension

    return labels


def load_labels(path, ext):
    """
    Load labels from a file based on the file extension.

    Args:
        path (str): The path to the file.
        ext (str): The file extension.

    Returns:
        numpy.ndarray: The loaded labels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if ext == ".mrc":
        with mrcfile.open(path, "r") as f:
            labels = f.data
    if ext == ".h5":
        with h5py.File(path, "r") as f:
            labels = f["labels/mitochondria"][:]
            if "raw" in f:
                print("h5 file already has image data")
    return labels


def run_correction(raw_input_path, label_input_path, output_path, fname, orig_label_path=None):
    continue_correction = True

    with mrcfile.open(raw_input_path, "r") as f:
        raw = f.data
    labels = load_labels(label_input_path, os.path.splitext(label_input_path)[1])
    labels.setflags(write=1)

    if orig_label_path is not None:
        orig_labels = load_labels(orig_label_path, os.path.splitext(orig_label_path)[1])
    else:
        orig_labels = None

    v = napari.Viewer()

    # if orig_labels is None:
    #     orig_labels = labels

    v.add_image(raw)
    v.add_labels(labels)
    if orig_labels is not None:
        v.add_labels(orig_labels)
    labels_layer = v.layers["labels"]
    labels_layer.mode = "paint"
    
    v.title = f"Tomo: {fname}, mitochondria"

    @magicgui(call_button="Paint New Vesicle [f]")
    def paint_new_mitos(v: napari.Viewer):
        layer = v.layers["labels"]
        layer.selected_label = 1
        layer.mode = "paint"

    @magicgui(call_button="Save Correction")
    def save_correction(v: napari.Viewer):
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        labels = v.layers["labels"].data
        raw = v.layers["raw"].data
        # labels = labels #label(labels)
        with h5py.File(output_path, "a") as f:
            ds = f.require_dataset(
                "labels/mitochondria", shape=labels.shape, dtype=labels.dtype, compression="gzip"
            )
            ds[:] = labels
            ds_raw = f.require_dataset(
                "raw", shape=raw.shape, dtype=raw.dtype, compression="gzip"
            )
            ds_raw[:] = raw
        show_info(f"Saved segmentation to {output_path}.")
    
    @magicgui(call_button="Stop Correction")
    def stop_correction(v: napari.Viewer):
        nonlocal continue_correction
        show_info("Stop correction.")
        continue_correction = False

    v.window.add_dock_widget(paint_new_mitos)
    v.window.add_dock_widget(save_correction)
    v.window.add_dock_widget(stop_correction)

    v.bind_key("s", lambda _:  save_correction(v))
    v.bind_key("q", lambda _:  stop_correction(v))
    v.bind_key("f", lambda _:  paint_new_mitos(v))

    napari.run()

    return continue_correction


def correct_mitochondria(args):
    base_path = args.base_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    raw_files = sorted(glob(os.path.join(base_path, "**/*raw.mrc"), recursive=True))
    label_files = sorted(glob(os.path.join(base_path, "**/*labels.mrc"), recursive=True))
    
    iteration = 0
    for raw_path, label_path in tqdm(zip(raw_files, label_files)):
        orig_label_path = None
        iteration += 1
        path, fname = os.path.split(raw_path)
        fname, _ = os.path.splitext(fname)
        fname = fname.replace("_raw", "")
        fname = fname + ".h5"
        # print("\nfname: ", fname)
        output_path = os.path.join(save_dir, fname)
        if iteration <= 0:
            continue

        if os.path.exists(output_path):
            # continue
            # orig_label_path = label_path
            label_path = output_path
        print(f"Loading: \n{raw_path} \n{label_path}\n")

        if not run_correction(raw_path, label_path, output_path, fname, orig_label_path):
            break


# def refine_mitochondria(args):
#     h5_file_paths = sorted(glob(os.path.join(args.base_path, "**/*.h5"), recursive=True))
#     for h5_file_path in tqdm(h5_file_paths):
#         path, fname = os.path.split(h5_file_path)
#         fname, _ = os.path.splitext(fname)
#         if not run_correction(h5_file_path, h5_file_path, path, fname):
#             break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=BASE_PATH, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Path to save the data to")
    args = parser.parse_args()

    correct_mitochondria(args)
    # refine_mitochondria(args)


if __name__ == "__main__":
    main()