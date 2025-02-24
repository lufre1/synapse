from elf.io import open_file
import argparse
from glob import glob
import os
import h5py
import mrcfile


def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)

def get_file_paths(path, ext=".mrc", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def main(base_path: str, ext: str = ".mrc", scale: int = 1, export_path=None):
    paths = get_file_paths(base_path, ext)
    for path in paths:
        with mrcfile.open(path) as mrc:
            print("\nMRC orig voxel size", mrc.voxel_size)
        print(path)
        # if "36859_J1_STEM750_66K_SP_06" not in path:
        #     continue
        with open_file(path, mode="r") as f:
            data = {}
            if ".mrc" in path or ".rec" in path:
                ndim = f["data"].ndim
                print("MRC File:", f["data"][:].shape)
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
                if scale != 1:
                    print("Size after downsampling", data["raw"].shape)
            else:
                print(f"Could not load file:\n{path}")
        output_path = os.path.join(export_path, os.path.basename(path).replace(".mrc", ".h5"))
        if os.path.exists(output_path):
            print("Output path already exist; skipping:", output_path)
        else:
            export_to_h5(data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="/home/freckmann15/data/mitochondria/cooper/fidi/20250212_test_I/20250212_test_I")
    parser.add_argument("--export_path", "-ep", type=str, default="/home/freckmann15/data/mitochondria/cooper/fidi/20250212_test_I/h5_export/")
    parser.add_argument("--ext", "-e", type=str, default=".mrc")
    parser.add_argument("--scale", "-s", type=int, default=1)
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    scale = args.scale
    export_path = args.export_path
    main(path, ext, scale, export_path)
