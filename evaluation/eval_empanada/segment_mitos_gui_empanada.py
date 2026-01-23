import os
import glob
import h5py
import napari

# directory containing .h5 files
# H5_DIR = os.environ.get("H5_DIR", ".")  # or set a path string here
H5_DIR = "/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_for_mitonet"

def iter_datasets(h5file):
    """Yield (path_in_file, dataset) for all datasets in the HDF5 file."""
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))
    datasets = []
    h5file.visititems(visitor)
    return datasets


def main():
    h5_paths = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in: {H5_DIR}")

    viewer = napari.Viewer()

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            for dset_name, dset in iter_datasets(f):
                data = dset[()]  # load into memory
                layer_name = f"{os.path.basename(h5_path)}::{dset_name}"
                viewer.add_image(data, name=layer_name)
            napari.run()


if __name__ == "__main__":
    main()