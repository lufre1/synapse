import argparse
from  torch_em.data.datasets.electron_microscopy.cellmap import get_cellmap_paths
# from elf.io import open_file


def main(args):
    path = args.path
    if path is None:
        get_cellmap_paths(path="",
                                 resolution=args.resolution,
                                 padding=0,
                                 download=True)
    else:
        if args.crop_id is None:
            get_cellmap_paths(path=path, padding=0, download=True)
        else:
            for crop_id in args.crop_id:
                get_cellmap_paths(path=path, crops=[crop_id], padding=0, download=True)
            # for p in paths:
            #     if f"{args.crop_id}" in p:
            #         get_cellmap_paths(path=path, padding=0, download=True) 
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
    argsparse.add_argument("--crop_id", "-c", type=int, nargs="+",default=None, help="Crop id(s), if given only download the crop(s)")
    args = argsparse.parse_args()
    main(args)
