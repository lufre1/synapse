import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import synapse.util as util
import elf.parallel as parallel
from elf.io import open_file


def main(args):
    paths = util.get_data_paths(args.input, args.file_extension)

    rows = []
    for path in tqdm(paths, total=len(paths)):
        with open_file(path, "r") as f:
            data = f[args.dataset_key][...]

        mask = np.isin(data, args.ids)

        # connected components
        cc = parallel.label(mask, block_shape=(128, 256, 256), verbose=False)

        # component sizes; exclude background (label 0)
        ids, counts = np.unique(cc, return_counts=True)
        counts = counts[ids != 0]

        if counts.size == 0:
            row = dict(
                file_path=path,
                amount_instances=0,
                smallest=np.nan,
                biggest=np.nan,
                std=np.nan,
                median=np.nan,
            )
        else:
            row = dict(
                file_path=path,
                amount_instances=int(counts.size),
                smallest=int(counts.min()),
                biggest=int(counts.max()),
                std=float(counts.std(ddof=0)),
                median=float(np.median(counts)),
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    output = args.output
    if output is None:
        output = os.path.join(args.input, f"instance_size_stats_{str(args.ids)}.csv")

    df.to_csv(output, index=False)
    print("Wrote:", output)


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--input", "-i", type=str, default="/mnt/lustre-grete/projects/nim00020/data/volume-em/cellmaps")
    argsparse.add_argument("--file_extension", "-e", type=str, default=".h5")
    argsparse.add_argument("--dataset_key", "-k", type=str, default="label_crop/all")
    argsparse.add_argument("--output", "-o", type=str, default=None)
    argsparse.add_argument("--ids", "-id", type=int, nargs='+', default=[3, 4, 5, 50])
    args = argsparse.parse_args()
    main(args)