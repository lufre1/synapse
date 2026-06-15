#!/usr/bin/env python
"""Comprehensive regression test for synapse/segment/ refactor.

Run from repo root:   python test_segment_imports.py

Categories:
  1. Import regression      - submodules, flat API, backward compat, cycles
  2. Smoke tests            - call moved functions with tiny in-memory arrays
                               (no GPU, no models, no network files)
  3. Circular dependency    - all segment modules imported together
"""
import os
import sys
import traceback

# ensure we test the local repo, not any installed version
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)


# ----------------------------------------------------------------------------
# 1. IMPORT REGRESSION
# ----------------------------------------------------------------------------


def test_segment_submodules_import():
    """Each submodule can be imported on its own."""
    from synapse.segment.mito import (
        segment_mitos,
        segment_mitos_ooc_wrapped,
        segment_mitos_ooc_optimized,
    )
    from synapse.segment.axons import segment_axons, segment_axons_ooc
    from synapse.segment.postprocessing import (
        refine_seg,
        compute_bboxes_ooc,
        postprocess_seg_3d_ooc,
        iterate_blocks,
    )
    print("  submodules (mito/axons/postprocessing) import OK")


def test_flat_api_imports():
    """All 20 names exported from synapse.segment.__init__."""
    from synapse.segment import (
        segment_mitos,
        segment_axons,
        segment_mitos_ooc_wrapped,
        segment_mitos_ooc_optimized,
        segment_axons_ooc,
        filter_segmentation,
        filter_small_objects,
        refine_seg,
        compute_bboxes_ooc,
        postprocess_seg_3d_ooc,
        adjust_size,
        downsample_to_shape,
        upsample_data,
        convert_white_patches_to_black,
        iterate_blocks,
        export_data,
        export_ooc_to_h5,
        get_3d_model,
        run_prediction,
        ZarrChannelWrapper,
    )
    assert all(callable(v) for v in (
        segment_mitos, segment_mitos_ooc_wrapped, segment_mitos_ooc_optimized,
        segment_axons, segment_axons_ooc,
        filter_segmentation, filter_small_objects, refine_seg,
        compute_bboxes_ooc, postprocess_seg_3d_ooc,
        adjust_size, downsample_to_shape, upsample_data,
        convert_white_patches_to_black, iterate_blocks,
        export_data, export_ooc_to_h5,
        get_3d_model, run_prediction, ZarrChannelWrapper,
    )), "Not all flat-API names are callable"
    print("  flat API -- all 20 names OK")


def test_backward_compat():
    """import synapse.util works (old callers do this)."""
    import synapse.util as util
    assert hasattr(util, "export_data")
    assert hasattr(util, "create_directories_if_not_exists")
    assert hasattr(util, "get_data_paths")
    assert hasattr(util, "get_3d_model")
    assert hasattr(util, "standardize_channel")
    print("  synapse.util backward compat OK")


def test_prediction_module():
    """synapse.prediction (never touched) still works -- main cyc-dep trap."""
    from synapse.prediction import (
        get_prediction_torch_em,
        run_prediction,
        get_3d_model,
    )
    print("  synapse.prediction OK")


def test_cristae_module():
    """synapse.cristae.segment (never touched) still importable."""
    from synapse.cristae.segment import run_cristae_segmentation
    assert callable(run_cristae_segmentation)
    print("  synapse.cristae.segment OK")


# ----------------------------------------------------------------------------
# 2. SMOKE TESTS -- no GPU, no models, pure numpy
# ----------------------------------------------------------------------------

import numpy as np


def test_filter_segmentation():
    """filter_segmentation on a tiny labeled array."""
    from synapse.segment import filter_segmentation

    seg = np.zeros((10, 10, 10), dtype=np.uint32)
    seg[0, 2:8, 2:8] = 1     # label 1, 36 voxels
    seg[2, 2:8, 2:8] = 2     # label 2, 36 voxels
    seg[4, :, :] = 3         # label 3, 100 voxels

    result = filter_segmentation(seg)
    assert result.shape == seg.shape
    assert result.dtype == seg.dtype
    for lbl in (1, 2, 3):
        assert (result == lbl).any(), f"Label {lbl} disappeared"
    print("  filter_segmentation (in-core CC) OK")


def test_filter_small_objects():
    from synapse.segment import filter_small_objects

    seg = np.ones((20, 20, 20), dtype=np.uint32)
    seg[0, 0, 0] = 99          # tiny object, size 1
    result = filter_small_objects(seg, min_size=10)
    assert result[0, 0, 0] == 0, "tiny object should have been removed"
    print("  filter_small_objects OK")


def test_downsample_to_shape():
    from synapse.segment import downsample_to_shape, upsample_data

    arr = np.random.random((64, 64, 64)).astype(np.float32)
    small = downsample_to_shape(arr, (32, 32, 32))
    assert small.shape == (32, 32, 32)

    big = upsample_data(small, factor=2)
    assert big.shape == (64, 64, 64)
    # nearest-neighbor round-trip is imperfect by design; verify shapes
    print("  downsample/upsample round-trip OK")


def test_adjust_size():
    from synapse.segment import adjust_size

    arr = np.random.random((20, 20, 20)).astype(np.float32)

    # rescale
    down = adjust_size(arr, scale=0.5, is_segmentation=True)
    assert down.shape == (10, 10, 10)

    # resize to target
    target = adjust_size(arr, orig_shape=(8, 8, 8), is_segmentation=True)
    assert target.shape == (8, 8, 8)
    print("  adjust_size (rescale + resize) OK")


def test_convert_white_patches_to_black_2d():
    from synapse.segment import convert_white_patches_to_black

    img = np.zeros((20, 20), dtype=np.uint8)
    img[0:5, 0:5] = 255               # 25-voxel patch
    img[10:15, 10:15] = 255
    result = convert_white_patches_to_black(img, min_patch_size=20)
    assert (result == 255).sum() == 0
    print("  convert_white_patches_to_black (2D) OK")


def test_convert_white_patches_to_black_3d():
    from synapse.segment import convert_white_patches_to_black

    img = np.zeros((10, 10, 10), dtype=np.uint8)
    img[0:3, 0:3, 0:3] = 255          # 27-voxel patch -> removed
    img[5, 5, 5] = 255                # 1-voxel patch  -> kept
    result = convert_white_patches_to_black(img, min_patch_size=20)
    assert (result == 255).sum() == 1, "only 1-voxel should remain"
    print("  convert_white_patches_to_black (3D) OK")


def test_postprocessing_ooc_callable():
    """OOC functions are importable; signature-check only (need zarr to call)."""
    from synapse.segment import (
        apply_size_filter_ooc,
        apply_size_filter_ooc_optim,
        compute_bboxes_ooc,
        postprocess_seg_3d_ooc,
    )
    for fn in (apply_size_filter_ooc, apply_size_filter_ooc_optim,
               compute_bboxes_ooc, postprocess_seg_3d_ooc):
        assert callable(fn)
    print("  postprocessing OOC functions -- importable OK")


# ----------------------------------------------------------------------------
# 3. CIRCULAR DEPENDENCY DETECTION
# ----------------------------------------------------------------------------


def test_no_circular_import():
    """All segment modules imported together in one pass.
    If any creates a cycle, the import fails."""
    from synapse.segment import segment_mitos
    from synapse.segment.mito import segment_mitos_ooc_wrapped
    from synapse.segment.axons import segment_axons, segment_axons_ooc
    from synapse.segment.postprocessing import (
        filter_segmentation, filter_small_objects, refine_seg,
        apply_size_filter_ooc, compute_bboxes_ooc,
    )
    from synapse.util import (
        export_data, iterate_blocks, ZarrChannelWrapper, get_3d_model,
    )
    from synapse.prediction import get_prediction_torch_em
    from synapse.cristae.segment import run_cristae_segmentation
    print("  circular dependency check -- no cycles OK")


# ----------------------------------------------------------------------------
# TEST RUNNER
# ----------------------------------------------------------------------------

_TESTS = [
    # 1. imports
    test_segment_submodules_import,
    test_flat_api_imports,
    test_backward_compat,
    test_prediction_module,
    test_cristae_module,
    # 3. circular dep
    test_no_circular_import,
    # 2. smoke (no GPU / no models)
    test_filter_segmentation,
    test_filter_small_objects,
    test_downsample_to_shape,
    test_adjust_size,
    test_convert_white_patches_to_black_2d,
    test_convert_white_patches_to_black_3d,
    test_postprocessing_ooc_callable,
]


def main():
    header = "=  synapse/segment/ -- import + smoke regression test  ="
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)
    print()

    failures = []
    for fn in _TESTS:
        name = fn.__name__.replace("test_", "")
        try:
            fn()
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL {name}")

    print()
    passed = len(_TESTS) - len(failures)
    print(f"Results: {passed}/{len(_TESTS)} passed")
    if failures:
        print()
        print("FAILURES:")
        for name, tb in failures:
            print("\n  -- " + name + " --")
            for line in tb.splitlines():
                print("    " + line)
        sys.exit(1)
    print("  All tests passed")


if __name__ == "__main__":
    main()
