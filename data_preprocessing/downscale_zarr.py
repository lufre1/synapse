import argparse
from typing import Optional, Sequence, Tuple, Union
import zarr
import numpy as np
from skimage.transform import rescale, resize
from tqdm import tqdm
from numcodecs import Blosc
import math


# ---------------------------------------------------------------------------
# OME-NGFF 0.4 metadata helpers
# ---------------------------------------------------------------------------

_AXES_ZYX = [
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _read_voxel_size(root: zarr.Group, key: str) -> Optional[Tuple[float, ...]]:
    """Return voxel_size for *key* from OME-NGFF multiscales or legacy root attrs, or None."""
    attrs = dict(root.attrs)
    # OME-NGFF multiscales
    for ms in attrs.get("multiscales", []):
        for ds in ms.get("datasets", []):
            if ds["path"] == key:
                for ct in ds.get("coordinateTransformations", []):
                    if ct["type"] == "scale":
                        return tuple(ct["scale"])
    # Legacy per-dataset attrs (e.g. s0/.zattrs)
    if key in root:
        ds_attrs = dict(root[key].attrs)
        if "voxel_size" in ds_attrs:
            return tuple(ds_attrs["voxel_size"])
    # Legacy root-level voxel_size — assumed to match the first / finest scale
    if "voxel_size" in attrs:
        return tuple(attrs["voxel_size"])
    return None


def _update_multiscales(
    root: zarr.Group,
    input_key: str,
    output_key: str,
    input_voxel_size: Tuple[float, ...],
    scale: Tuple[float, ...],
    unit: str = "micrometer",
) -> None:
    """Write/update OME-NGFF 0.4 multiscales entry for input and output datasets.

    output voxel size = input voxel size / scale_factor
    (downscaling by 0.5 doubles the physical voxel size).
    """
    out_voxel = tuple(v / s for v, s in zip(input_voxel_size, scale))

    axes = [{"name": ax["name"], "type": ax["type"], "unit": unit} for ax in _AXES_ZYX]

    def _ds_entry(path, voxel):
        return {
            "path": path,
            "coordinateTransformations": [{"type": "scale", "scale": list(voxel)}],
        }

    attrs = dict(root.attrs)
    if "multiscales" in attrs:
        ms = attrs["multiscales"][0]
        ms.setdefault("axes", axes)
        datasets = ms.get("datasets", [])
        # Ensure input_key is present
        if not any(d["path"] == input_key for d in datasets):
            datasets.append(_ds_entry(input_key, input_voxel_size))
        # Replace or append output_key
        datasets = [d for d in datasets if d["path"] != output_key]
        datasets.append(_ds_entry(output_key, out_voxel))
        datasets.sort(key=lambda d: d["path"])
        ms["datasets"] = datasets
    else:
        attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": axes,
                "datasets": sorted(
                    [_ds_entry(input_key, input_voxel_size), _ds_entry(output_key, out_voxel)],
                    key=lambda d: d["path"],
                ),
                "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(input_voxel_size)}],
            }
        ]

    root.attrs.update(attrs)
    print(
        f"OME-NGFF multiscales updated: '{input_key}' {list(input_voxel_size)} → "
        f"'{output_key}' {[round(v, 8) for v in out_voxel]} {unit}"
    )


def resize_zarr_dataset_safe(
    zarr_path: str,
    input_key: str,
    output_key: str,
    scale_factor: Union[float, Sequence[float]],
    is_segmentation: bool = False,
    output_chunks: Tuple[int, ...] = None,
    output_chunk_shape: Tuple[int, ...] = None,
    halo: Union[int, Sequence[int], None] = None,
    order_raw: int = 1,
    overwrite: bool = True,
    voxel_size: Optional[Tuple[float, ...]] = None,
    unit: str = "micrometer",
):
    """
    Safely resize a 2D/3D Zarr dataset by iterating over OUTPUT chunks.

    This is safer than independently rescaling input blocks and stitching them,
    because each output chunk is computed from the corresponding source region.

    Parameters
    ----------
    zarr_path:
        Path to the Zarr store.
    input_key:
        Input dataset key.
    output_key:
        Output dataset key.
    scale_factor:
        Scalar or per-axis scale factor.
        Example:
            0.5        -> downscale all axes by 2
            (1, 0.5, 0.5) -> keep Z, downscale Y/X by 2
            (1, 2, 2)  -> keep Z, upscale Y/X by 2
    is_segmentation:
        If True, uses nearest-neighbor interpolation and no anti-aliasing.
    output_chunks / output_chunk_shape:
        Chunk shape for the output dataset. Use either name.
        If None, a reasonable default is derived from the input chunks and scale.
    halo:
        Optional source halo in source voxels/pixels for raw interpolation.
        If None, a default halo is chosen:
            - segmentation: 0
            - raw: 2 for order 1, 4 for order >= 3
    order_raw:
        Interpolation order for raw data. Typical values:
            1 = linear
            3 = cubic
    overwrite:
        Whether to overwrite output_key if it exists.
    """
    root = zarr.open(zarr_path, mode="a")
    if input_key not in root:
        raise KeyError(f"Dataset '{input_key}' not found in {zarr_path}")

    data_in = root[input_key]
    in_shape = tuple(data_in.shape)
    ndim = data_in.ndim

    if ndim not in (2, 3):
        raise ValueError(f"Only 2D/3D arrays are supported, got ndim={ndim}")

    # Normalize scale factor
    if isinstance(scale_factor, (int, float)):
        scale = (float(scale_factor),) * ndim
    else:
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) == 1:
            scale = (scale_factor[0],) * ndim
        elif len(scale_factor) == ndim:
            scale = scale_factor
        else:
            raise ValueError(f"scale_factor must have length 1 or {ndim}, got {scale_factor}")

    if any(s <= 0 for s in scale):
        raise ValueError(f"All scale factors must be > 0, got {scale}")

    out_shape = tuple(max(1, int(round(s * f))) for s, f in zip(in_shape, scale))

    # Choose interpolation behavior
    if is_segmentation:
        order = 0
        anti_aliasing = False
    else:
        order = order_raw
        # anti_aliasing only matters for downscaling
        anti_aliasing = any(sf < 1.0 for sf in scale)

    # Halo
    if halo is None:
        if is_segmentation:
            halo_tup = (0,) * ndim
        else:
            if order <= 1:
                halo_tup = (2,) * ndim
            else:
                halo_tup = (4,) * ndim
    elif isinstance(halo, int):
        halo_tup = (halo,) * ndim
    else:
        halo_tup = tuple(int(h) for h in halo)
        if len(halo_tup) != ndim:
            raise ValueError(f"halo must have length {ndim}, got {halo_tup}")

    # Output chunking
    if output_chunk_shape is not None and output_chunks is not None:
        raise ValueError("Use only one of output_chunk_shape or output_chunks")
    if output_chunk_shape is not None:
        output_chunks = output_chunk_shape

    if output_chunks is None:
        if getattr(data_in, "chunks", None) is not None:
            # Derive output chunks from input chunks and scale
            output_chunks = tuple(
                max(1, min(out_shape[d], int(round(data_in.chunks[d] * scale[d]))))
                for d in range(ndim)
            )
        else:
            output_chunks = tuple(min(s, 64) for s in out_shape)
    else:
        output_chunks = tuple(int(c) for c in output_chunks)
        if len(output_chunks) != ndim:
            raise ValueError(f"output_chunks must have length {ndim}, got {output_chunks}")
    # get compressor
    if data_in.compressor is not None:
        compressor = data_in.compressor
    else:
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    # Create output dataset
    data_out = root.require_dataset(
        output_key,
        shape=out_shape,
        chunks=output_chunks,
        dtype=data_in.dtype,
        compressor=compressor,
        overwrite=overwrite,
    )

    print(f"Input shape:  {in_shape}")
    print(f"Output shape: {out_shape}")
    print(f"Scale:        {scale}")
    print(f"Output chunks:{output_chunks}")
    print(f"Halo:         {halo_tup}")
    print(f"Mode:         {'segmentation' if is_segmentation else 'raw'}")

    # Number of chunks per axis
    grid_shape = tuple(math.ceil(out_shape[d] / output_chunks[d]) for d in range(ndim))
    n_chunks = int(np.prod(grid_shape))

    def chunk_slices_from_index(flat_idx: int):
        coords = []
        rem = flat_idx
        for g in reversed(grid_shape):
            coords.append(rem % g)
            rem //= g
        coords = tuple(reversed(coords))

        out_sl = []
        for d, c in enumerate(coords):
            start = c * output_chunks[d]
            stop = min(start + output_chunks[d], out_shape[d])
            out_sl.append(slice(start, stop))
        return tuple(out_sl)

    for chunk_id in tqdm(range(n_chunks), desc="Resizing output chunks"):
        out_sl = chunk_slices_from_index(chunk_id)
        out_chunk_shape = tuple(sl.stop - sl.start for sl in out_sl)

        # Map output chunk to source coordinate interval.
        # out index range [a, b) corresponds roughly to source [a/scale, b/scale)
        src_start_f = [sl.start / scale[d] for d, sl in enumerate(out_sl)]
        src_stop_f = [sl.stop / scale[d] for d, sl in enumerate(out_sl)]

        # Expand by halo for safer interpolation context.
        src_read_start = []
        src_read_stop = []
        for d in range(ndim):
            rs = max(0, int(math.floor(src_start_f[d])) - halo_tup[d])
            re = min(in_shape[d], int(math.ceil(src_stop_f[d])) + halo_tup[d])
            if re <= rs:
                re = min(in_shape[d], rs + 1)
            src_read_start.append(rs)
            src_read_stop.append(re)

        src_sl = tuple(slice(s, e) for s, e in zip(src_read_start, src_read_stop))
        src = data_in[src_sl]

        # Compute where the desired output chunk sits inside the resized source crop.
        crop_scale = scale
        tmp_shape = tuple(
            max(1, int(round((src_read_stop[d] - src_read_start[d]) * crop_scale[d])))
            for d in range(ndim)
        )

        if src.size == 0:
            out_chunk = np.zeros(out_chunk_shape, dtype=data_in.dtype)
            data_out[out_sl] = out_chunk
            continue

        # Fast paths for masks/constant blocks
        if is_segmentation:
            if np.all(src == 0):
                data_out[out_sl] = np.zeros(out_chunk_shape, dtype=data_in.dtype)
                continue
            if np.all(src == src.flat[0]):
                data_out[out_sl] = np.full(out_chunk_shape, src.flat[0], dtype=data_in.dtype)
                continue

        # Resize source crop to temporary target crop
        resized_tmp = resize(
            src,
            output_shape=tmp_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )

        # Map exact output chunk position into the temporary resized crop
        tmp_crop = []
        for d in range(ndim):
            rel_start_src = src_start_f[d] - src_read_start[d]
            rel_stop_src = src_stop_f[d] - src_read_start[d]

            ts = int(round(rel_start_src * scale[d]))
            te = int(round(rel_stop_src * scale[d]))

            ts = max(0, min(ts, resized_tmp.shape[d]))
            te = max(ts + 1, min(te, resized_tmp.shape[d]))

            # Ensure exact requested output size if rounding differs
            wanted = out_chunk_shape[d]
            actual = te - ts
            if actual > wanted:
                te = ts + wanted
            elif actual < wanted:
                ts = max(0, te - wanted)
                te = min(resized_tmp.shape[d], ts + wanted)
                ts = te - wanted

            tmp_crop.append(slice(ts, te))

        out_chunk = resized_tmp[tuple(tmp_crop)]

        # Final shape guard
        if out_chunk.shape != out_chunk_shape:
            # fallback: force exact shape
            out_chunk = resize(
                out_chunk,
                output_shape=out_chunk_shape,
                order=order,
                preserve_range=True,
                anti_aliasing=False if is_segmentation else anti_aliasing,
            )

        if is_segmentation:
            out_chunk = out_chunk.astype(data_in.dtype, copy=False)
        else:
            if np.issubdtype(data_in.dtype, np.integer):
                info = np.iinfo(data_in.dtype)
                out_chunk = np.clip(out_chunk, info.min, info.max)
            out_chunk = out_chunk.astype(data_in.dtype, copy=False)

        data_out[out_sl] = out_chunk

    print(f"Successfully wrote '{output_key}' to {zarr_path}")

    # --- OME-NGFF metadata ---
    effective_voxel_size = voxel_size or _read_voxel_size(root, input_key)
    if effective_voxel_size is not None and len(effective_voxel_size) == ndim:
        _update_multiscales(root, input_key, output_key, effective_voxel_size, scale, unit=unit)
    else:
        print(
            "No voxel_size found — skipping OME-NGFF metadata update. "
            "Pass voxel_size= or add legacy voxel_size to root .zattrs to enable it."
        )


def downscale_zarr_dataset(zarr_path, input_key="0", output_key="1", scale_factor=0.5, z_block_size=64,
                           args=None):
    print(f"Opening Zarr store at: {zarr_path}")
    root = zarr.open(zarr_path, mode='a')
    
    if input_key not in root:
        raise KeyError(f"Dataset '{input_key}' not found in the Zarr store.")
        
    data_in = root[input_key]
    old_shape = data_in.shape
    ndim = len(old_shape)
    
    # Calculate new shape (e.g., dividing Z, Y, X by 2)
    # 1. Normalize scale_factor to a tuple matching the array dimensions
    if isinstance(scale_factor, (int, float)):
        # If single number, apply to all dimensions
        scale_tuple = (float(scale_factor),) * ndim
    elif len(scale_factor) == 1:
        # If list of one number, apply to all dimensions
        scale_tuple = (float(scale_factor),) * ndim
    elif len(scale_factor) == ndim:
        # If list/tuple matches dimensions, use as is
        scale_tuple = tuple(float(s) for s in scale_factor)
        print("scaling factors (zyx):", scale_tuple)
    else:
        raise ValueError(f"scale_factor must be a single value or match dimensions ({ndim}), got {scale_factor}")

    new_shape = tuple(int(s * f) for s, f in zip(old_shape, scale_tuple))

    print(f"Original shape: {old_shape}")
    print(f"New target shape: {new_shape}")
    
    # Create the new dataset '1'
    # We reuse the chunks and compressor from the original dataset for consistency
    data_out = root.require_dataset(
        output_key, 
        shape=new_shape, 
        chunks=data_in.chunks, 
        dtype=data_in.dtype,
        compressor=data_in.compressor,
        overwrite=True
    )
    
    # Process chunk-by-chunk along the Z-axis to avoid OOM errors
    print(f"Rescaling in Z-blocks of {z_block_size} slices...")
    
    for z in tqdm(range(0, old_shape[0], z_block_size)):
        z_end = min(z + z_block_size, old_shape[0])
        
        # Load just this block into RAM
        block = data_in[z:z_end]
        
        # Rescale the block
        # preserve_range=True prevents skimage from normalizing values to 0.0 - 1.0
        # anti_aliasing=True is recommended for raw image data to prevent artifacts
        if not args.is_segmentation:
            rescaled_block = rescale(
                block, 
                scale=scale_factor, 
                preserve_range=True, 
                anti_aliasing=True,
                channel_axis=None # Explicitly state this is a spatial 3D volume, not channels
            )
        else:
            rescaled_block = rescale(
                block, 
                scale=scale_factor, 
                preserve_range=True, 
                anti_aliasing=False, # MUST be False for masks/labels
                order=0,             # MUST be 0 (nearest-neighbor) for masks/labels
                channel_axis=None 
            )
        
        # Convert back to the original datatype (rescale outputs floats by default)
        rescaled_block = rescaled_block.astype(data_in.dtype)
        
        # Calculate the corresponding Z-indices in the downscaled dataset
        z_out_start = int(z * scale_tuple[0])
        z_out_end_expected = z_out_start + rescaled_block.shape[0]
        
        # Clamp the end index so we don't exceed the new total shape
        z_out_end = min(z_out_end_expected, new_shape[0])
        
        # Calculate how many slices we are actually allowed to write
        valid_z_len = z_out_end - z_out_start
        
        # Write the rescaled block to disk, slicing off any extra frame at the end
        data_out[z_out_start:z_out_end] = rescaled_block[:valid_z_len]

    print(f"Successfully created dataset '{output_key}'!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", "-i", required=True, type=str, help="Path to Zarr file")
    p.add_argument("--input_key", "-k", default=None)
    p.add_argument("--output_key", "-ok", type=str, default=None)
    p.add_argument("--scale", "-s", type=float, nargs="+", default=[0.5], help="zyx downscale factor")
    p.add_argument("--is_segmentation", "-is", action="store_true", default=False)
    p.add_argument("--z_chunked", "-zc", action="store_true", default=False, help="old approach, might reprocduce inconsitent output shapes!")
    p.add_argument(
        "--voxel_size", "-vs", type=float, nargs="+", default=None,
        help="Input voxel size in zyx order (e.g. --voxel_size 0.025 0.005 0.005). "
             "If omitted the script reads it from existing zarr metadata.",
    )
    p.add_argument("--unit", "-u", type=str, default="micrometer", help="Physical unit for voxel_size (default: micrometer)")
    args = p.parse_args()
    file = args.input_path
    voxel_size = tuple(args.voxel_size) if args.voxel_size else None
    if args.z_chunked:
        downscale_zarr_dataset(
            file,
            z_block_size=64,
            input_key=args.input_key,
            output_key=args.output_key,
            scale_factor=args.scale,
            args=args,
        )
    else:
        resize_zarr_dataset_safe(
            zarr_path=file,
            input_key=args.input_key,
            output_key=args.output_key,
            scale_factor=args.scale,
            is_segmentation=args.is_segmentation,
            output_chunk_shape=None,
            overwrite=True,
            voxel_size=voxel_size,
            unit=args.unit,
        )