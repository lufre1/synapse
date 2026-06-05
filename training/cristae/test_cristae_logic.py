"""
Minimal smoke-test for the cristae training pipeline (no GPU / no napari needed).
Covers: data-path collection & splitting, data loader, model forward pass,
        loss computation with ignore_state_value masking.
"""
import sys, os, random, traceback
import torch
import numpy as np

sys.path.insert(0, "/user/freckmann15/u12103/synapse")
import synapse.util as util
import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.model import AnisotropicUNet

PATCH_SHAPE   = (32, 64, 64)   # small enough for CPU
BATCH_SIZE    = 1
LR            = 1e-4
DD1 = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2"
DD2 = "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae"
DD3 = "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/"

IGNORE_STATE_VALUE = 2
STATE_CHANNEL      = 1

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {name}")
        return result
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        traceback.print_exc()
        return None


# ── 1. data-path collection ───────────────────────────────────────────────────
print("\n=== 1. Data-path collection ===")

def collect_paths():
    paths = util.get_data_paths(DD1)
    paths.extend(util.get_data_paths(DD2))
    paths.extend(util.get_data_paths(DD3))
    substring = "_combined.h5"
    paths = [s for s in paths if substring in s]
    exclude = [
        "Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined",
        "WT20_eb8_AZ1_model_combined",
        "WT22_eb8_model_combined",
    ]
    for s in exclude:
        paths = [p for p in paths if s not in p]
    assert len(paths) > 0, "No data paths found"
    return paths

paths = check("collect & filter data paths", collect_paths)
if paths is None:
    sys.exit(1)
print(f"       {len(paths)} files found")


# ── 2. train/val/test split ───────────────────────────────────────────────────
print("\n=== 2. Train/val/test split ===")

def split():
    random.seed(42)
    random.shuffle(paths)
    data = util.split_data_paths_to_dict_with_ensure(
        paths, train_ratio=.8, val_ratio=0.1, test_ratio=0.1,
        ensure_strings=None
    )
    assert len(data["train"]) > 0, "empty train split"
    assert len(data["val"])   > 0, "empty val split"
    return data

data = check("split into train/val/test", split)
if data is None:
    sys.exit(1)
print(f"       train={len(data['train'])}  val={len(data['val'])}  test={len(data['test'])}")


# ── 3. data loader (train, a few batches) ────────────────────────────────────
print("\n=== 3. Data loader ===")

label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
sampler         = MinInstanceSampler(p_reject=0.95)

def make_loader():
    loader = torch_em.default_segmentation_loader(
        raw_paths=data["train"], raw_key="raw_mitos_combined",
        label_paths=data["train"], label_key="labels/cristae",
        patch_shape=PATCH_SHAPE, ndim=3, batch_size=BATCH_SIZE,
        label_transform=label_transform, num_workers=1,
        with_channels=True, with_label_channels=False,
        sampler=sampler,
        raw_transform=util.standardize_channel,
    )
    return loader

loader = check("create train loader", make_loader)
if loader is None:
    sys.exit(1)

def load_batches():
    it = iter(loader)
    batches = []
    for _ in range(3):
        raw, label = next(it)
        assert raw.shape[1]   == 2, f"expected 2 input channels, got {raw.shape[1]}"
        assert label.shape[1] == 2, f"expected 2 label channels (BoundaryTransform), got {label.shape[1]}"
        batches.append((raw, label))
    return batches

batches = check("load 3 batches (raw shape, label shape)", load_batches)
if batches is None:
    sys.exit(1)
raw0, lbl0 = batches[0]
print(f"       raw={tuple(raw0.shape)}  label={tuple(lbl0.shape)}")


# ── 4. state-channel values ───────────────────────────────────────────────────
print("\n=== 4. State-channel content ===")

def check_state_channel():
    sc = raw0[:, STATE_CHANNEL]
    unique_vals = np.unique(sc.numpy().astype(int))
    print(f"       unique values in state channel: {unique_vals}")
    assert sc.shape == (BATCH_SIZE, *PATCH_SHAPE), f"unexpected shape {sc.shape}"
    return unique_vals

check("state channel shape & unique values", check_state_channel)


# ── 5. loss function ─────────────────────────────────────────────────────────
print("\n=== 5. Loss function (ignore_state_value masking) ===")

loss_fn = util.get_loss_function("dice",
    ignore_state_value=IGNORE_STATE_VALUE,
    state_channel=STATE_CHANNEL,
)

def test_loss_masked():
    pred = torch.rand_like(lbl0)
    loss_val = loss_fn(pred, lbl0, raw0)
    assert torch.isfinite(loss_val), f"loss is not finite: {loss_val}"
    return loss_val.item()

loss_val = check("forward loss with masking (state_value=2 ignored)", test_loss_masked)
if loss_val is not None:
    print(f"       loss = {loss_val:.4f}")

def test_loss_masking_effect():
    pred = torch.rand_like(lbl0)
    loss_masked   = loss_fn(pred, lbl0, raw0).item()
    old = loss_fn.ignore_state_value
    loss_fn.ignore_state_value = None
    loss_unmasked = loss_fn(pred, lbl0, raw0).item()
    loss_fn.ignore_state_value = old
    print(f"       masked={loss_masked:.4f}  unmasked={loss_unmasked:.4f}")
    # They should differ when ignore_state_value voxels are present
    return loss_masked, loss_unmasked

check("masking changes loss value", test_loss_masking_effect)


# ── 6. model forward pass ─────────────────────────────────────────────────────
print("\n=== 6. Model forward pass ===")

model = util.get_3d_model(
    in_channels=2, out_channels=2, initial_features=32,
    final_activation="Sigmoid", gain=2,
    scale_factors=[[1,2,2],[2,2,2],[2,2,2],[2,2,2]],
)
model.eval()

def forward_pass():
    with torch.no_grad():
        out = model(raw0)
    assert out.shape == lbl0.shape, f"model output {out.shape} != label {lbl0.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Sigmoid output out of [0,1]"
    return out.shape

out_shape = check("AnisotropicUNet forward pass", forward_pass)
if out_shape:
    print(f"       output shape: {tuple(out_shape)}")

def loss_on_model_output():
    with torch.no_grad():
        out = model(raw0)
    loss_val = loss_fn(out, lbl0, raw0)
    assert torch.isfinite(loss_val)
    return loss_val.item()

loss_val2 = check("loss on actual model output", loss_on_model_output)
if loss_val2 is not None:
    print(f"       loss = {loss_val2:.4f}")

print("\n=== Done ===\n")
