"""Probe the largest batch size that fits on the current GPU for the cristae 3D UNet.

ONE job, in-process sweep — no submit→queue→OOM→repeat loop. Builds the *identical* model used by
`train_cristae.py` (`util.get_3d_model`, 2->2 channels, feature_size 32, Sigmoid, anisotropic scale
factors) and runs a few fwd+bwd+optimizer steps on synthetic tensors at each batch size, ascending,
until it OOMs. Memory is dominated by activations (batch x precision), so two regimes are swept:

  - fp32 : autocast OFF  -> matches flashoptim's "single precision" footprint (the heavier one).
  - amp  : autocast ON   -> matches the standard mixed_precision=True trainer.

Optimizer-state memory (plain AdamW fp32 moments) is <~0.2 GB for this model — negligible vs the
multi-GB activation maps — so this faithfully answers "what batch fits" without touching FlashAdamW.

Run via the config (recommended, pins the 80 GB card):
  python sbatch_runner.py configs/training/cristae/cristae_probe_maxbatch.yaml
or directly on a GPU node:
  python training/cristae/probe_max_batch.py --batch_sizes 8,12,16,20,24,28,32,40,48
"""
import argparse
import sys

import torch

import synapse.util as util  # same import as train_cristae.py:20


# Exactly train_cristae.py:322-331,412-415 (norm defaults to None, matching the cristae checkpoints).
IN_CHANNELS, OUT_CHANNELS = 2, 2
GAIN = 2
FINAL_ACTIVATION = "Sigmoid"
SCALE_FACTORS = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


def _is_oom(err: Exception) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def _try_batch(model, optimizer, batch_size, patch_shape, amp, iters, device):
    """Run `iters` train steps at this batch size; return peak GB on success, raise on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.randn(batch_size, IN_CHANNELS, *patch_shape, device=device)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            out = model(x)
            # Surrogate loss: activation memory (what OOMs) is identical regardless of the real loss,
            # and this avoids masked-Dice NaN edge cases on random inputs.
            loss = out.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del x, out, loss
    return peak_gb


def sweep(model, batch_sizes, patch_shape, regime, iters, device):
    amp = regime == "amp"
    print(f"\n===== regime: {regime} (autocast={'ON' if amp else 'OFF'}) =====", flush=True)
    print(f"{'batch':>6} | {'peak GB':>8} | result", flush=True)
    print("-" * 32, flush=True)
    max_fit = None
    for bs in batch_sizes:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        try:
            peak_gb = _try_batch(model, optimizer, bs, patch_shape, amp, iters, device)
            print(f"{bs:>6} | {peak_gb:>8.2f} | FIT", flush=True)
            max_fit = bs
        except Exception as err:  # noqa: BLE001 - we re-raise non-OOM errors
            if not _is_oom(err):
                raise
            print(f"{bs:>6} | {'-':>8} | OOM (stop; larger sizes also OOM)", flush=True)
            del optimizer
            torch.cuda.empty_cache()
            break
        del optimizer
        torch.cuda.empty_cache()
    return max_fit


def main():
    parser = argparse.ArgumentParser(description="Probe max batch size for the cristae 3D UNet.")
    parser.add_argument("--batch_sizes", type=str, default="8,12,16,20,24,28,32,40,48",
                        help="Comma-separated batch sizes to try, ascending.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 256, 256],
                        help="Patch shape (z y x).")
    parser.add_argument("--feature_size", type=int, default=32,
                        help="Initial feature size of the 3D UNet (must match training).")
    parser.add_argument("--regimes", type=str, default="fp32,amp",
                        help="Comma-separated precision regimes to sweep: fp32 (flashoptim) and/or amp.")
    parser.add_argument("--iters", type=int, default=3,
                        help="Train steps per batch size (peak is reached within 1-2).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: probe_max_batch.py requires CUDA (no GPU detected).", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    batch_sizes = sorted(int(b) for b in args.batch_sizes.split(","))
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    bad = [r for r in regimes if r not in ("fp32", "amp")]
    if bad:
        print(f"ERROR: unknown regime(s) {bad}; allowed: fp32, amp", file=sys.stderr)
        sys.exit(1)
    patch_shape = tuple(args.patch_shape)

    props = torch.cuda.get_device_properties(0)
    print("=" * 60, flush=True)
    print(f"GPU: {props.name}  total memory: {props.total_memory / 1e9:.1f} GB", flush=True)
    print(f"patch_shape={patch_shape}  feature_size={args.feature_size}  "
          f"in/out channels={IN_CHANNELS}/{OUT_CHANNELS}  iters/batch={args.iters}", flush=True)
    print(f"batch sizes: {batch_sizes}   regimes: {regimes}", flush=True)
    print("=" * 60, flush=True)

    model = util.get_3d_model(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, initial_features=args.feature_size,
        final_activation=FINAL_ACTIVATION, gain=GAIN, scale_factors=SCALE_FACTORS,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f} M  "
          f"(fp32 weights+grads ~{2 * n_params * 4 / 1e9:.2f} GB, AdamW states ~{2 * n_params * 4 / 1e9:.2f} GB)",
          flush=True)

    results = {}
    for regime in regimes:
        results[regime] = sweep(model, batch_sizes, patch_shape, regime, args.iters, device)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY — max batch size that fits", flush=True)
    for regime in regimes:
        mf = results[regime]
        print(f"  {regime:>5}: {mf if mf is not None else 'NONE (even smallest OOM)'}", flush=True)
    fp32_max = results.get("fp32")
    if fp32_max is not None:
        print(f"\n-> For the flashoptim config (runs fp32/single-precision), use batch_size = {fp32_max} "
              f"or one step lower for headroom (validation + cuDNN workspace + fragmentation).", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
