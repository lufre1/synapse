import torch
from torch_em.util import load_model
import argparse
import os


def export_model(args):
    if os.path.isdir(args.checkpoint_path):
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.dirname(args.checkpoint_path)

    # Load and save on the requested device. Default "cpu" makes the exported file portable:
    # a CPU-tagged model loads on CPU-only machines (no map_location needed) and still moves to
    # GPU via .to(device). Saving while the model sits on cuda bakes "cuda:0" into the pickle, which
    # then fails to torch.load on a CPU-only host.
    model = load_model(checkpoint_path, device=args.device)
    model = model.to(args.device).eval()
    print(f"Loaded model from {args.checkpoint_path} on {args.device}")

    # Resolve the export path: a ".pt" path is used as the file; anything else is treated as a
    # directory (created if needed) that receives "model.pt" — matching the exported_models/*/model.pt layout.
    export_path = args.export_path
    if export_path.endswith(".pt"):
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    else:
        os.makedirs(export_path, exist_ok=True)
        export_path = os.path.join(export_path, "model.pt")

    torch.save(model, export_path)
    print("Successfully exported model to", export_path)

    # Sanity check: confirm the file deserializes on CPU (what a CPU-only laptop / the UI would do).
    torch.load(export_path, map_location="cpu", weights_only=False)
    print("Verified: reloads on CPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained torch_em model to a portable .pt file")
    parser.add_argument("--checkpoint_path", "-cp", required=True, type=str, default="", help="Path to the directory where 'best.pt' resides.")
    parser.add_argument("--export_path", "-ep", required=True, type=str, default="", help="Output directory (receives model.pt) or a .pt file path.")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        help="Device to load/save on. 'cpu' (default) makes the export portable; "
                             "use 'cuda' only if you specifically want a GPU-pinned file.")
    args = parser.parse_args()

    export_model(args)
