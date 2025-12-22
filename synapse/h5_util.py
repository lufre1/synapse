from pathlib import Path
from typing import Optional, Tuple, Union
import h5py
import numpy as np


def read_voxel_size(
    h5_path: Union[str, Path],
    h5_key: str = "raw",
    *,
    default: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float]:
    """
    Read the ``voxel_size`` attribute and return it as (z, y, x).

    Parameters
    ----------
    h5_path : str or pathlib.Path
        Path to the HDF5 file.
    h5_key : str
        Internal HDF5 path to the object that holds the attribute,
        e.g. ``"/raw/volume0"`` or ``"/raw"``.
    default : tuple of three floats, optional
        Value to return if the attribute is missing.  If ``None`` (default)
        a ``KeyError`` is raised.

    Returns
    -------
    voxel_size : tuple (z, y, x)
        The voxel size in the order you asked for.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the key or the attribute is missing and ``default`` is ``None``.
    TypeError
        If the attribute cannot be interpreted as a length‑3 float array.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Open the file (read‑only).  The context manager guarantees close.
    # ------------------------------------------------------------------
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # ------------------------------------------------------------------
        # 2️⃣  Locate the object (dataset or group) that should contain the attr.
        # ------------------------------------------------------------------
        if h5_key not in f:
            raise KeyError(f"Key '{h5_key}' not found in file '{h5_path}'.")
        obj = f[h5_key]                     # could be a Dataset or a Group

        # ------------------------------------------------------------------
        # 3️⃣  Pull the attribute – h5py returns a NumPy scalar/array.
        # ------------------------------------------------------------------
        if "voxel_size" not in obj.attrs:
            if default is not None:
                # Return the user‑provided fallback (already in z,y,x order)
                return tuple(default)
            raise KeyError(
                f"'voxel_size' attribute missing on object '{h5_key}'."
            )

        raw = obj.attrs["voxel_size"]

        # ------------------------------------------------------------------
        # 4️⃣  Normalise to a plain (x, y, z) NumPy array first.
        # ------------------------------------------------------------------
        # The attribute may be:
        #   * a plain 1‑D array   → shape (3,)
        #   * a structured array  → dtype=[('x',f4),('y',f4),('z',f4)]
        #   * a Python list/tuple → automatically converted by h5py
        #   * a scalar (unlikely) → we guard against it
        # We coerce everything to a flat ``np.ndarray`` of shape (3,).
        if isinstance(raw, np.ndarray):
            if raw.dtype.names:                     # structured array case
                # Preserve the order defined by the field names.
                # We explicitly read x, y, z regardless of the order in the file.
                xyz = np.array([float(raw[name]) for name in ("x", "y", "z")],
                               dtype=np.float32)
            else:
                # Plain numeric array – must be length‑3
                if raw.shape != (3,):
                    raise TypeError(
                        f"'voxel_size' attribute on '{h5_key}' has shape {raw.shape}; "
                        "expected a length‑3 array."
                    )
                xyz = raw.astype(np.float32)
        else:
            # scalar or Python sequence – try to cast to a length‑3 tuple
            try:
                seq = tuple(float(v) for v in raw)   # works for list/tuple
                if len(seq) != 3:
                    raise ValueError
                xyz = np.array(seq, dtype=np.float32)
            except Exception as exc:
                raise TypeError(
                    f"Unable to interpret 'voxel_size' attribute on '{h5_key}' "
                    "as a length‑3 float sequence."
                ) from exc

        # ------------------------------------------------------------------
        # 5️⃣  Re‑order to (z, y, x) and return as a plain Python tuple.
        # ------------------------------------------------------------------
        # xyz = [x, y, z] → we need [z, y, x]
        zyx = (float(xyz[2]), float(xyz[1]), float(xyz[0]))
        return zyx


def read_h5(path, key, scale_factor=1, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == ("prediction" or "pred" in key) and not ("foreground" in key or "boundar" in key) :
                image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            else:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            print(f"{key} data shape after downsampling during read operation", image.shape)
            # if not key == "raw":
            #     print(np.unique(image))

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys
