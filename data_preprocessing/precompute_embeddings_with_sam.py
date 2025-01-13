import os
import torch
from tqdm import tqdm
import argparse
from elf.io import open_file
from glob import glob
from micro_sam.util import precompute_image_embeddings, get_sam_model, get_device


def compute_embeddings(path, output_path):
    filename = os.path.basename(path)
    out_filenpath = os.path.join(output_path, filename)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = get_sam_model(device=device)
    with open_file(path, "r") as f:
        raw = f["raw"][:]
        precompute_image_embeddings(predictor, input_=raw, save_path=out_filenpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/trimmed_all", help="Path to the root data directory")
    parser.add_argument("--output_path", "-o",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/embeddings", help="Path to the output data directory")
    args = parser.parse_args()
    base_path = args.base_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    for path in tqdm(paths):
        compute_embeddings(path, output_path)
