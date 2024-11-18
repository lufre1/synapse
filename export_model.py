import torch
from torch_em.util import load_model
import argparse


def export_model(args):
    model = load_model(args.checkpoint_path)
    print("Loaded model from", args.checkpoint_path)
    torch.save(model, args.export_path)
    print("Successfully exported model to", args.export_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--checkpoint_path", "-cp", required=True, type=str, default="", help="Path to the directory where 'best.pt' resides.")
    parser.add_argument("--export_path", "-ep", required=True, type=str, default="", help="Path to the directory plus the name of the exported model.")
    args = parser.parse_args()

    export_model(args)
