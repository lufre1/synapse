import h5py
import napari

from skimage.measure import label
from skimage.segmentation import watershed


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
    raw_file = "36859_J1_STEM750_66K_SP_01_rec_2kb1dawbp_crop.h5"
    print("Load raw ...")
    with h5py.File(raw_file, "r") as f:
        raw = f["raw"][:]

    pred_file = "mito-net_latest_02-07-24_prediction_36859_J1_STEM750_66K_SP_01_rec_2kb1dawbp_crop.h5"
    print("Load pred ...")
    with h5py.File(pred_file, "r") as f:
        pred = f["prediction"][:]
    fg, bd = pred

    print("Run segmentation ...")
    seg = run_watershed(fg, bd)

    print("Visualize ...")
    check_predictions(raw, fg, bd, seg)


if __name__ == "__main__":
    main()
