import torch
from torch_em.util import load_model
import argparse
import os


def export_model(args):
    if os.path.isdir(args.checkpoint_path):
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.dirname(args.checkpoint_path)
    model = load_model(checkpoint_path)
    print("Loaded model from", args.checkpoint_path)
    if os.path.isdir(args.export_path):
        export_path = os.path.join(args.export_path, "model.pt")
    else:
        export_path = args.export_path
    torch.save(model, export_path)
    print("Successfully exported model to", args.export_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--checkpoint_path", "-cp", required=True, type=str, default="", help="Path to the directory where 'best.pt' resides.")
    parser.add_argument("--export_path", "-ep", required=True, type=str, default="", help="Path to the directory plus the name of the exported model.")
    args = parser.parse_args()

    export_model(args)
