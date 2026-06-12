"""All segmentation entry points for synapse.

Import from here for a clean, flat namespace::

    from synapse.segment import segment_mitos, segment_axons, refine_seg, iterate_blocks

Individual modules are also importable::

    from synapse.segment.mito import segment_mitos_ooc_wrapped
    from synapse.segment.axons import segment_axons_ooc
    from synapse.segment.postprocessing import filter_segmentation, apply_size_filter_ooc
"""

# -- main seg entry points --
from synapse.segment.mito import (
    segment_mitos,
    segment_mitos_ooc_wrapped,
    segment_mitos_ooc_optimized,
)
from synapse.segment.axons import (
    segment_axons,
    segment_axons_ooc,
)
from synapse.segment.postprocessing import (
    apply_size_filter_ooc,
    apply_size_filter_ooc_optim,
    filter_segmentation,
    filter_small_objects,
    refine_seg,
    compute_bboxes_ooc,
    postprocess_seg_3d_ooc,
    adjust_size,
    downsample_to_shape,
    upsample_data,
    convert_white_patches_to_black,
)

# -- shared infra kept in synapse/util.py --
from synapse.util import (
    export_data,
    export_ooc_to_h5,
    get_3d_model,
    run_prediction,
    ZarrChannelWrapper,
    iterate_blocks,
)
