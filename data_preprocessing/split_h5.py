import argparse
from glob import glob
import os
from tqdm import tqdm
import synapse.util as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/fidi_2025/exported_to_hdf5/m13dko/37371_O4_66K_TS_SC_50_rec_2Kb1dawbp_crop_F.h5", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/home/freckmann15/data/mitochondria/cooper/fidi_2025/exported_to_hdf5/m13dko/", help="Path to the export directory")
    parser.add_argument("--split_after_z_slices", "-s", type=int, default=101, help="Split after z slices")
    args = parser.parse_args()
    base_path = args.base_path
    export_path = args.export_path

    if os.path.isdir(base_path):
        paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    elif os.path.isfile(base_path):
        paths = [base_path]
    else:
        raise ValueError(f"Invalid base path: {base_path}")

    for path in tqdm(paths, desc="Processing files..."):
        output_path = os.path.join(export_path, os.path.basename(path).replace(".h5", "_1.h5"))
        output_path2 = os.path.join(export_path, os.path.basename(path).replace(".h5", "_2.h5"))
        if os.path.exists(output_path):
            print("output path already exists:", output_path)
            continue
        data = {}
        data = util.read_data(path)
        split1, split2 = {}, {}
        for key in data:
            split1[key] = data[key][:args.split_after_z_slices]
            split2[key] = data[key][args.split_after_z_slices:]
        print("split1 keys", split1.keys())
        print("split2 keys", split2.keys())
        util.export_data(output_path, split1)
        util.export_data(output_path2, split2)


if __name__ == "__main__":
    main()