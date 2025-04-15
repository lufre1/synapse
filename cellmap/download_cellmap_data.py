import argparse
from  torch_em.data.datasets.electron_microscopy.cellmap import get_cellmap_paths
# from elf.io import open_file


def main(args):
    path = args.path
    if path is None:
        path = get_cellmap_paths(path="",
                                 resolution=args.resolution,
                                 padding=0,
                                 download=True)
    else:
        path = get_cellmap_paths(path=path, padding=0, download=True)
    # import napari
    # with open_file(path, "r") as f:
    #     print(f.keys())
    #     v = napari.Viewer()
    #     for k in f.keys():
    #         if "label" in k and "mito" in k:
    #             v.add_labels(f[k])
    #         elif "raw" in k:
    #             v.add_image(f[k])


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--path", "-p", type=str, default="/scratch-grete/projects/nim00007/data/cellmap/")
    argsparse.add_argument("--resolution", "-r", type=str, default="s0")
    args = argsparse.parse_args()
    main(args)
