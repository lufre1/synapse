import argparse
import fsspec, zarr
import dask.array as da  # we import dask to help us manage parallel access to the big dataset


def find_s0_dataset(group, base_path=""):
    """ Recursively find the full-resolution 's0' dataset in the given Zarr group. """
    for k, v in group.items():
        path = f"{base_path}/{k}" if base_path else k
        if isinstance(v, zarr.Group):
            result = find_s0_dataset(v, path)  # Recurse into subgroups
            if result:
                return result
        elif isinstance(v, zarr.Array) and k == "s0":
            return path  # Return full path to s0 dataset
    return None


def find_all_s0_datasets(group, base_path=""):
    """ Recursively find all 's0' datasets in the given Zarr group. """
    s0_datasets = {}

    for k, v in group.items():
        path = f"{base_path}/{k}" if base_path else k
        if isinstance(v, zarr.Group):
            s0_datasets.update(find_all_s0_datasets(v, path))  # Recurse into subgroups
        elif isinstance(v, zarr.Array) and k == "s0":
            s0_datasets[path] = v  # Store dataset path and array reference

    return s0_datasets


def load_s0_datasets(zarr_root):
    """ Load all 's0' datasets under 'em' and 'labels' groups into Dask arrays. """
    all_s0_datasets = {}

    for group_name in ["em", "labels"]:  # Search in both 'em' and 'labels'
        if group_name in zarr_root:
            s0_datasets = find_all_s0_datasets(zarr_root[group_name])
            for path, zdata in s0_datasets.items():
                all_s0_datasets[path] = da.from_array(zdata, chunks=zdata.chunks)  # Convert to Dask array

    return all_s0_datasets


def download(args):
    # group = zarr.open(zarr.N5FSStore('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5', anon=True)) # access the root of the n5 container
    group = zarr.open(zarr.N5FSStore(args.uri, anon=True))
    # breakpoint()
    # zdata = group['em/fibsem-uint16/s0'] # s0 is the the full-resolution data for this particular volume
    # s0_raw_path = find_s0_dataset(group)
    
    # if s0_raw_path:
    #     zdata = group[s0_raw_path]
    # else:
    #     print("could not find S0 dataset")
    #     return
    
    # ddata = da.from_array(zdata, chunks=zdata.chunks)
    # print("ddata shape", ddata.shape)
    # print("ddata", ddata)
    # result = ddata[0].compute() # get the first slice of the data as a numpy array
    # print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Data from Janelia")
    parser.add_argument("--uri", "-u", type=str,
                        default="s3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5",
                        required=False, help="Path to the data")
    parser.add_argument("--output_path", "-o", type=str, required=False, help="Output path to save the data")
    args = parser.parse_args()
    download(args)
