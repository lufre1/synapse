"""Repair a flashoptim (FlashAdamW + bf16) cristae checkpoint into a loadable serialized model.

FlashOptimTrainer saves checkpoints with an EMPTY `init` and bfloat16 `model_state`, so the standard
torch_em loader (`get_trainer`, which rebuilds the model from `init["model_kwargs"]`) cannot
reconstruct it. This rebuilds the known cristae architecture via `synapse.util.get_3d_model` (exactly
how train_cristae.py builds it), loads the flashoptim weights cast to fp32, and saves a SERIALIZED
`nn.Module`.

Point the eval config's `model_path` at the produced file (named `model.pt`, NOT `best.pt`):
synapse-net's `get_prediction_torch_em` then `torch.load`s a non-dir model_path directly, bypassing
`get_trainer` / `init` entirely.
"""
import argparse
import os
from collections import OrderedDict

import torch

import synapse.util as util  # same model builder train_cristae.py uses


def build_cristae_model():
    # Mirrors train_cristae.py:412-415 (in/out=2, feat=32, Sigmoid, gain=2, anisotropic scale factors).
    return util.get_3d_model(
        in_channels=2, out_channels=2, initial_features=32,
        final_activation="Sigmoid", gain=2,
        scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    )


def repair(src_ckpt, out_file):
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict"))
    if state is None:
        raise KeyError(f"no model_state in {src_ckpt}")
    prefix = "_orig_mod."  # in case the model was torch.compile'd
    state = OrderedDict(
        (k[len(prefix):] if k.startswith(prefix) else k, v.float()) for k, v in state.items()
    )
    model = build_cristae_model()
    model.load_state_dict(state)  # strict: fail loudly if the arch does not match
    model.eval()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    torch.save(model, out_file)
    print(f"repaired: {src_ckpt}\n      -> {out_file}  "
          f"(epoch={ck.get('epoch')}, best_epoch={ck.get('best_epoch')}, "
          f"best_metric={ck.get('best_metric')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flashoptim checkpoint DIR containing <name>.pt")
    ap.add_argument("--out", required=True, help="output DIR for the repaired serialized model.pt")
    ap.add_argument("--name", default="best", help="checkpoint name to repair (best|latest)")
    args = ap.parse_args()
    repair(os.path.join(args.src, f"{args.name}.pt"), os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
