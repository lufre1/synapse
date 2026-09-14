"""
Debug script: mirrors the debug block in train_axons_volem.py but runs on GPU
with a downsized patch to fit in 10 GB VRAM.  No napari / no trainer.
"""
import random
import torch
import torch_em
import torch_em.loss
from torch_em.data import MinInstanceSampler

import synapse.util as util
import synapse.h5_util as h5_util

# ── config ──────────────────────────────────────────────────────────────────
DD  = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/all_cutouts_s2_new"
SDD = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented_s2_new_white_removed/"
TDD = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split_s2_new/"

PATCH_SHAPE  = (64, 128, 128)   # reduced from 64×512×512
BATCH_SIZE   = 1
LR           = 1e-4
N_DEBUG      = 5                # batches for initial stats
N_STEPS      = 40               # gradient steps to watch
DEVICE       = "cuda"

# ── data ─────────────────────────────────────────────────────────────────────
data_paths = util.get_data_paths(DD)
data_paths += util.get_data_paths(SDD)
data_paths += util.get_data_paths(TDD)
data_paths = [p for p in data_paths if "labels/axons" in h5_util.get_all_keys_from_h5(p)]

random.seed(42)
random.shuffle(data_paths)
data = util.split_data_paths_to_dict_with_ensure(data_paths, ensure_strings=("4007", "4009"))
print(f"train={len(data['train'])}  val={len(data['val'])}")

sampler = MinInstanceSampler(p_reject=0.95)
loader = torch_em.default_segmentation_loader(
    raw_paths=data["train"], raw_key="raw",
    label_paths=data["train"], label_key="labels/axons",
    patch_shape=PATCH_SHAPE, ndim=3, batch_size=BATCH_SIZE,
    raw_transform=torch_em.transform.raw.normalize_percentile,
    label_transform=torch_em.transform.labels_to_binary,
    num_workers=4,
    with_channels=False, with_label_channels=False,
    sampler=sampler, n_samples=200,
)

# ── model ────────────────────────────────────────────────────────────────────
model = util.get_3d_model(
    out_channels=1, in_channels=1,
    scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features=32,
    norm=None,
    final_activation="Sigmoid",
).to(DEVICE)

dice_bce_loss = torch_em.loss.CombinedLoss(
    torch_em.loss.DiceLoss(),
    torch.nn.BCELoss(),
)
dice_only = torch_em.loss.DiceLoss()

print(f"\n{'='*60}")
print(f"Patch shape : {PATCH_SHAPE}  BS={BATCH_SIZE}")
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM total  : {vram:.1f} GB")
print(f"{'='*60}")

# ── initial batch stats ───────────────────────────────────────────────────────
print(f"\n--- Initial batch stats ({N_DEBUG} batches, untrained model) ---")
model.eval()
it = iter(loader)
for i in range(N_DEBUG):
    raw, label = next(it)
    raw, label = raw.to(DEVICE), label.to(DEVICE)
    with torch.no_grad():
        pred = model(raw)

    # label may be [B,1,D,H,W] or [B,D,H,W]
    label_f = label.float()
    if label_f.ndim < pred.ndim:
        label_f = label_f.unsqueeze(1)

    loss_val  = dice_bce_loss(pred, label_f)
    dice_val  = dice_only(pred, label_f)
    fg_frac   = label_f.mean().item()
    print(
        f"  batch {i}: "
        f"raw=[{raw.min():.3f},{raw.max():.3f}] mean={raw.mean():.3f} | "
        f"label shape={tuple(label.shape)} fg={fg_frac:.4f} | "
        f"pred=[{pred.min():.3f},{pred.max():.3f}] mean={pred.mean():.3f} | "
        f"dice_bce={loss_val.item():.4f}  dice={dice_val.item():.4f}"
    )
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

print(f"\nPeak VRAM during forward pass: {vram_used:.2f} GB")

# ── gradient-step simulation ──────────────────────────────────────────────────
print(f"\n--- Gradient-step simulation ({N_STEPS} steps, dice+bce loss) ---")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
it2 = iter(loader)

dice_history = []
loss_history = []
pred_mean_history = []

for step in range(N_STEPS):
    raw_s, label_s = next(it2)
    raw_s, label_s = raw_s.to(DEVICE), label_s.to(DEVICE)

    label_sf = label_s.float()
    if label_sf.ndim < 5:
        label_sf = label_sf.unsqueeze(1)

    optimizer.zero_grad()
    pred_s = model(raw_s)
    loss_s = dice_bce_loss(pred_s, label_sf)
    loss_s.backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()

    dice_val = dice_only(pred_s.detach(), label_sf).item()
    pred_mean = pred_s.detach().mean().item()
    fg = label_sf.mean().item()

    dice_history.append(dice_val)
    loss_history.append(loss_s.item())
    pred_mean_history.append(pred_mean)

    print(
        f"  step {step:3d}: loss={loss_s.item():.4f}  dice={dice_val:.4f}  "
        f"pred_mean={pred_mean:.4f}  fg={fg:.4f}  |grad|={grad_norm:.2f}"
    )

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"  Dice first 5 steps : {[f'{v:.3f}' for v in dice_history[:5]]}")
print(f"  Dice last  5 steps : {[f'{v:.3f}' for v in dice_history[-5:]]}")
print(f"  Pred mean first 5  : {[f'{v:.3f}' for v in pred_mean_history[:5]]}")
print(f"  Pred mean last  5  : {[f'{v:.3f}' for v in pred_mean_history[-5:]]}")

delta_dice = dice_history[0] - dice_history[-1]
print(f"\n  Dice improvement (first→last): {delta_dice:+.4f}  ", end="")
if delta_dice > 0.05:
    print("✓ model is learning")
elif abs(delta_dice) < 0.01:
    print("⚠ STAGNANT — loss not moving")
else:
    print("? marginal movement")

# detect constant-output collapse
pred_spread = max(pred_mean_history) - min(pred_mean_history)
print(f"  Pred-mean spread over all steps: {pred_spread:.4f}  ", end="")
if pred_spread < 0.02:
    print("⚠ CONSTANT OUTPUT — likely collapsed")
else:
    print("✓ output is varying")

print(f"{'='*60}")
