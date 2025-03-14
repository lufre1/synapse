import h5py
import napari
import tifffile

from skimage.measure import label
from skimage.segmentation import watershed
import numpy as np


def check_predictions(raw, fg, bd, seg=None):
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(fg)
    v.add_image(bd)
    if seg is not None:
        v.add_labels(seg)
    napari.run()


def run_watershed(fg, bd):
    # Subtract boundaries from foreground to obtain non-touching objects.
    # Threshold it and apply connected components to obtain the seeds.
    object_threshold = 0.6
    seed_map = (fg - bd) > object_threshold
    seed_map = label(seed_map)

    # Note: this part only refines the objects, I am skipping it for now because my laptop
    # does not have enough memory.
    # Run a watershed with the thresholded foreground predictions to get the segmented objects.
    # foreground_threshold = 0.5
    # seg = watershed(bd, seed_map, mask=fg > foreground_threshold)

    return seed_map


def main():
    raw_file = "/home/freckmann15/data/lucchi/lucchi_test.h5"#"36859_J1_STEM750_66K_SP_01_rec_2kb1dawbp_crop.h5"
    print("Load raw ...")
    with h5py.File(raw_file, "r") as f:
        raw = f["raw"][:]

    pred_file = "/home/freckmann15/data/predictions/micro_sam_3d/lucchi_test.tif"#"mito-net_latest_02-07-24_prediction_36859_J1_STEM750_66K_SP_01_rec_2kb1dawbp_crop.h5"
    print("Load pred ...")
    pred = tifffile.imread(pred_file)
    print(pred.shape, "np.unique(pred)", np.unique(pred))
    v = napari.Viewer()
    v.add_image(pred)
    napari.run()
        #pred = f["prediction"][:]
    fg = np.where(pred == 1, 1, 0)
    bd = np.where(pred == 2, 1, 0)
    # fg = pred == 1
    # bd = pred == 2
    
    print("fg and bd shape and dtype", fg.shape, fg.dtype, bd.shape, bd.dtype)

    print("Run segmentation ...")
    seg = run_watershed(fg, bd)

    print("Visualize ...")
    check_predictions(raw, fg, bd, seg)


if __name__ == "__main__":
    main()
