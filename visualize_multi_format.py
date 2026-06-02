#!/usr/bin/env python3

"""
Visualization script for HDF5, Zarr, MRC, REC, and TIFF files with automatic
file discovery, naming scheme handling, and shape alignment.
"""

import argparse
import os
import numpy as np
import napari
from skimage.transform import resize
import h5py
import zarr
import tifffile
import mrcfile
import glob
from typing import List, Dict, Any, Tuple, Union

def get_file_paths(path: str, extensions: List[str] = None) -> List[str]:
    """Get list of file paths from a path (file or directory)."""
    # If it's a file, return it directly
    if os.path.isfile(path):
        return [path]
    
    # If it's a directory, find all matching files
    if extensions is None:
        extensions = [".h5", ".hdf5", ".zarr", ".mrc", ".rec", ".tif", ".tiff"]
    
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        
    # Also check for .zarr directories
    paths.extend(glob.glob(os.path.join(path, "**", "*.zarr"), recursive=True))
    
    return sorted(paths)

def detect_file_format(file_path: str) -> str:
    """Detect the file format based on extension."""
    _, ext = os.path.splitext(file_path.lower())
    
    if ext in [".h5", ".hdf5"]:
        return "hdf5"
    elif ext == ".zarr":
        return "zarr"
    elif ext == ".mrc":
        return "mrc"
    elif ext in [".rec"]:
        return "rec"
    elif ext in [".tif", ".tiff"]:
        return "tif"
    else:
        # Try to detect from file content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header == b'\x89HDF':
                    return "hdf5"
                elif header.startswith(b'ZARR'):
                    return "zarr"
        except:
            pass
        return "unknown"

def load_hdf5_data(file_path: str, key: str = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Load data from HDF5 file, returning all datasets if no key specified."""
    with h5py.File(file_path, 'r') as f:
        if key is not None:
            # Load specific dataset
            return f[key][:]
        else:
            # Extract all datasets recursively
            data = {}
            
            # Helper function to extract all datasets from groups
            def extract_data(group, data_dict, prefix=""):
                for key, item in group.items():
                    full_key = f"{prefix}/{key}" if prefix else key
                    if isinstance(item, h5py.Group):
                        # Recursively extract data from subgroups
                        extract_data(item, data_dict, prefix=full_key)
                    else:
                        # This is a dataset, load it
                        data_dict[full_key] = item[:]
            
            extract_data(f, data)
            return data

def load_zarr_data(file_path: str, key: str = None) -> np.ndarray:
    """Load data from Zarr file."""
    z = zarr.open(file_path, mode='r')
    if key is None:
        # Find first array
        for k in z.keys():
            if isinstance(z[k], zarr.Array):
                key = k
                break
    if key is None:
        raise ValueError(f"No data key found in {file_path}")
    return z[key][:]

def load_mrc_data(file_path: str, key: str = None) -> np.ndarray:
    """Load data from MRC file."""
    with mrcfile.open(file_path, permissive=True) as mrc:
        return mrc.data

def load_rec_data(file_path: str, key: str = None) -> np.ndarray:
    """Load data from REC file."""
    # REC files are typically binary, read with numpy
    return np.fromfile(file_path, dtype=np.int16)

def load_tif_data(file_path: str, key: str = None) -> np.ndarray:
    """Load data from TIFF file."""
    return tifffile.imread(file_path)

def load_data(file_path: str, key: str = None) -> np.ndarray:
    """Load data from any supported file format."""
    fmt = detect_file_format(file_path)
    
    loaders = {
        "hdf5": load_hdf5_data,
        "zarr": load_zarr_data,
        "mrc": load_mrc_data,
        "rec": load_rec_data,
        "tif": load_tif_data
    }
    
    if fmt in loaders:
        return loaders[fmt](file_path, key)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

def get_data_shape(data: np.ndarray) -> Tuple[int, ...]:
    """Get the shape of the data."""
    return data.shape

def align_shapes(datasets: List[np.ndarray]) -> List[np.ndarray]:
    """
    Align all datasets to the same shape using skimage.resize for proper resampling.
    """
    if len(datasets) <= 1:
        return datasets
    
    try:
        # Ensure all datasets have the same number of dimensions
        first_ndim = datasets[0].ndim
        if not all(d.ndim == first_ndim for d in datasets):
            print("Warning: Datasets have different dimensions, skipping alignment")
            return datasets
            
        # Find maximum shape across all datasets
        max_shape = tuple(max(dim[i] for dim in [d.shape for d in datasets]) 
                          for i in range(first_ndim))
        
        print(f"Aligning to max shape: {max_shape}")
        
        # Create new arrays with the maximum shape
        aligned_datasets = []
        
        for i, dataset in enumerate(datasets):
            current_shape = dataset.shape
            print(f"Dataset {i} shape: {current_shape}")
            
            # If already correct shape, no need to resize
            if current_shape == max_shape:
                aligned_datasets.append(dataset)
                print(f"  No resizing needed")
                continue
            
            # Use skimage.resize for proper resampling
            print(f"  Resizing from {current_shape} to {max_shape}")
            try:
                # Determine if this is segmentation data (discrete values) 
                # by checking if it's likely to be labeled data
                is_segmentation = False
                # Check if data looks like segmentation (integer dtype, small range, discrete values)
                if (dataset.dtype.kind in 'iu' and 
                    np.max(dataset) <= 255 and 
                    np.min(dataset) >= 0):
                    # Check for relatively few unique values (typical of segmentation)
                    unique_vals = len(np.unique(dataset))
                    if unique_vals < 1000:  # Not too many unique values for segmentation
                        is_segmentation = True
                        print("  Detected as segmentation data - using nearest neighbor interpolation")
                
                # Choose appropriate resizing parameters
                if is_segmentation:
                    # For segmentation, use nearest neighbor to preserve discrete values
                    resized = resize(dataset, max_shape, preserve_range=True, anti_aliasing=False, order=0)
                else:
                    # For continuous data (images), use linear interpolation
                    resized = resize(dataset, max_shape, preserve_range=True, anti_aliasing=False, order=1)
                
                if dataset.dtype == np.uint8 or dataset.dtype == np.uint16:
                    resized = resized.astype(dataset.dtype)
                aligned_datasets.append(resized)
                print(f"  Resized shape: {resized.shape}")
            except Exception as e:
                print(f"  Resize failed, falling back to cropping/padding: {e}")
                # Fall back to cropping/padding if resize fails
                aligned = np.zeros(max_shape, dtype=dataset.dtype)
                slices = tuple(slice(0, min(current_shape[i], max_shape[i])) 
                               for i in range(len(max_shape)))
                aligned[slices] = dataset[slices]
                aligned_datasets.append(aligned)
                print(f"  Cropped/padded shape: {aligned.shape}")
        
        return aligned_datasets
    except Exception as e:
        print(f"Error in align_shapes: {e}")
        import traceback
        traceback.print_exc()
        return datasets
    
    try:
        # Ensure all datasets have the same number of dimensions
        first_ndim = datasets[0].ndim
        if not all(d.ndim == first_ndim for d in datasets):
            print("Warning: Datasets have different dimensions, skipping alignment")
            return datasets
            
        # Find maximum shape across all datasets
        max_shape = tuple(max(dim[i] for dim in [d.shape for d in datasets]) 
                          for i in range(first_ndim))
        
        print(f"Aligning to max shape: {max_shape}")
        
        aligned_datasets = []
        
        for i, dataset in enumerate(datasets):
            current_shape = dataset.shape
            print(f"Dataset {i} shape: {current_shape}")
            
            # Create new array with maximum shape
            aligned = np.zeros(max_shape, dtype=dataset.dtype)
            
            # Calculate slices for copying
            slices = tuple(slice(0, min(current_shape[i], max_shape[i])) 
                           for i in range(len(max_shape)))
            
            print(f"Copying slices: {slices}")
            
            # Copy data
            aligned[slices] = dataset[slices]
            aligned_datasets.append(aligned)
            print(f"Aligned shape: {aligned.shape}")
        
        return aligned_datasets
    except Exception as e:
        print(f"Error in align_shapes: {e}")
        import traceback
        traceback.print_exc()
        return datasets

def visualize_aligned_datasets(datasets: Dict[str, np.ndarray], 
                             names: List[str] = None,
                             voxel_size: Tuple[float, ...] = None):
    """Visualize multiple datasets in Napari with aligned shapes."""
    if names is None:
        names = list(datasets.keys())
    
    viewer = napari.Viewer()
    
    # Add datasets to viewer
    for i, (name, data) in enumerate(datasets.items()):
        # Check if this should be treated as segmentation/labels
        is_segmentation = False
        
        # More robust segmentation detection
        name_lower = name.lower()
        
        # Check for common segmentation indicators
        if ('seg' in name_lower or 'label' in name_lower or 'mask' in name_lower or
            name_lower.endswith(('_seg', '_label', '_mask', '-seg', '-label', '-mask'))):
            is_segmentation = True
        # Also treat TIFF files as segmentation by default (based on file extension)
        # (This would be determined at load time, not here)
        
        # Additional check: if it's integer type and likely segmentation
        if not is_segmentation and data.dtype.kind in 'iu' and data.size > 0:
            # Check if it's likely segmentation data (few unique values, reasonable range)
            unique_vals = len(np.unique(data))
            if unique_vals < 1000 and np.max(data) <= 255 and np.min(data) >= 0:
                # Very rough heuristic: if mostly small integers, likely segmentation
                is_segmentation = True
        
        if is_segmentation:
            viewer.add_labels(data, name=name, scale=voxel_size)
        elif "raw" in name_lower or name == "0" or name_lower.endswith("_raw"):
            viewer.add_image(data, name=name, scale=voxel_size)
        else:
            # Default to image
            viewer.add_image(data, name=name, scale=voxel_size)
    
    # Bring raw layer to bottom if exists (so it doesn't cover up other layers)
    raw_layers = [layer for layer in viewer.layers if "raw" in layer.name.lower()]
    if raw_layers:
        # Move ALL raw layers to the bottom while maintaining their relative order
        for layer in reversed(raw_layers):
            viewer.layers.remove(layer)
            viewer.layers.insert(0, layer)
    
    return viewer

def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5, Zarr, MRC, REC, and TIFF files")
    parser.add_argument("--paths", "-p", nargs="+", required=True,
                       help="Paths to files or directories to visualize")
    parser.add_argument("--key", "-k", type=str, default=None,
                       help="Specific key to load from data files")
    parser.add_argument("--scale", "-s", type=float, default=1.0,
                       help="Scale factor for visualization")
    parser.add_argument("--voxel_size", "-vs", type=float, nargs=3, default=None,
                       help="Voxel size in nm (z, y, x)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="Directory to save aligned data (optional)")
    
    args = parser.parse_args()
    
    # Collect all file paths from all provided paths (files or directories)
    all_files = []
    for path in args.paths:
        # Direct file path
        if os.path.isfile(path):
            all_files.append(path)
        # Directory path - find all matching files
        elif os.path.isdir(path):
            files = get_file_paths(path)
            all_files.extend(files)
        else:
            # If path is neither file nor directory, try to find files matching the pattern
            if "*" in path or "?" in path or "[" in path:
                # It's a glob pattern
                found_files = glob.glob(path)
                all_files.extend(found_files)
            else:
                print(f"Warning: Path {path} does not exist as file or directory")
    
    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f"  {f}")
    
    # Load all datasets
    datasets = {}
    names = []
    
    # First, determine the target shape from the first file
    target_shape = None
    first_file_loaded = False
    
    # Track which files are TIFF files to mark them as segmentation
    tiff_files = set()
    
    for file_path in all_files:
        try:
            print(f"Loading {file_path}...")
            data = load_data(file_path, args.key)
            
            # Generate base name from file path
            basename = os.path.basename(file_path)
            name, _ = os.path.splitext(basename)
            
            # Track TIFF files
            if basename.lower().endswith(('.tif', '.tiff')):
                tiff_files.add(file_path)
            
            if isinstance(data, dict):
                # Multiple datasets returned from HDF5 file
                for dataset_name, dataset_data in data.items():
                    # Create unique name for each dataset
                    full_name = f"{name}_{dataset_name}"
                    # Replace forward slashes with underscores for valid naming
                    full_name = full_name.replace("/", "_")
                    
                    # For the first file, determine the target shape
                    if not first_file_loaded:
                        target_shape = dataset_data.shape
                        first_file_loaded = True
                        print(f"Target shape set from first file: {target_shape}")
                    
                    datasets[full_name] = dataset_data
                    names.append(full_name)
                    print(f"  Loaded {dataset_name} with shape: {dataset_data.shape}")
            else:
                # Single dataset
                # Handle duplicate names
                original_name = name
                counter = 1
                while name in datasets:
                    name = f"{original_name}_{counter}"
                    counter += 1
                
                # For the first file, determine the target shape
                if not first_file_loaded:
                    target_shape = data.shape
                    first_file_loaded = True
                    print(f"Target shape set from first file: {target_shape}")
                
                datasets[name] = data
                names.append(name)
                print(f"  Loaded shape: {data.shape}")
            
        except Exception as e:
            print(f"  Failed to load {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(datasets) == 0:
        print("No datasets were loaded successfully.")
        return
    
    print(f"\nLoaded {len(datasets)} datasets total:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")
    
    # If we have a target shape, align all datasets to it
    if target_shape is not None:
        print(f"\nAligning all datasets to target shape: {target_shape}")
        aligned_datasets = []
        aligned_dict = {}
        
        for name, data in datasets.items():
            # Check if this dataset should be treated as segmentation
            # (based on file extension or naming convention)
            is_segmentation = False
            
            # Check if this file was a TIFF file (which we treat as segmentation by default)
            # We need to trace back from dataset name to original file
            # Just be more aggressive about detecting what should be segmentation
            
            # Very robust segmentation detection
            name_lower = name.lower()
            
            # Explicit segmentation indicators
            if ('seg' in name_lower or 'label' in name_lower or 'mask' in name_lower or
                name_lower.endswith(('_seg', '_label', '_mask', '-seg', '-label', '-mask'))):
                is_segmentation = True
            # Also treat TIFF files as segmentation by default
            elif any(name_lower.endswith(ext) for ext in ['.tif', '.tiff']):
                is_segmentation = True
            # If it's integer type with reasonable range for segmentation
            elif data.dtype.kind in 'iu' and data.size > 0:
                # Check if it's likely segmentation data (few unique values, reasonable range)
                try:
                    unique_vals = len(np.unique(data))
                    max_val = np.max(data)
                    min_val = np.min(data)
                    if (unique_vals < 1000 and max_val <= 255 and min_val >= 0 and
                        # Additional sanity check for discrete labels
                        (max_val <= 100 or unique_vals <= 50)):
                        is_segmentation = True
                except:
                    pass  # fall back to normal behavior if unique calculation fails
            # If it's integer dtype with reasonable range for segmentation
            elif data.dtype.kind in 'iu' and data.size > 0:
                # Check if it's likely segmentation data (few unique values, reasonable range)
                unique_vals = len(np.unique(data))
                max_val = np.max(data)
                min_val = np.min(data)
                if (unique_vals < 1000 and max_val <= 255 and min_val >= 0 and
                    # Additional sanity check: likely discrete labels if mostly small integers
                    (max_val <= 100 or unique_vals <= 50)):
                    is_segmentation = True
            
            if data.shape == target_shape:
                # No alignment needed
                aligned_datasets.append(data)
                aligned_dict[name] = data
                print(f"  {name}: No alignment needed (already correct shape)")
            else:
                # Align to target shape using skimage.resize
                print(f"  {name}: Resizing from {data.shape} to {target_shape}")
                try:
                    # Choose appropriate resizing parameters
                    if is_segmentation:
                        # For segmentation, use nearest neighbor to preserve discrete values
                        print(f"  Using nearest neighbor interpolation for segmentation data")
                        resized = resize(data, target_shape, preserve_range=True, anti_aliasing=False, order=0)
                    else:
                        # For continuous data (images), use linear interpolation
                        print(f"  Using linear interpolation for image data")
                        resized = resize(data, target_shape, preserve_range=True, anti_aliasing=False, order=1)
                    
                    if data.dtype == np.uint8 or data.dtype == np.uint16:
                        resized = resized.astype(data.dtype)
                    aligned_datasets.append(resized)
                    aligned_dict[name] = resized
                    print(f"    Resized shape: {resized.shape}")
                except Exception as e:
                    print(f"  Resize failed, falling back to cropping/padding: {e}")
                    # Fall back to cropping/padding if resize fails
                    aligned = np.zeros(target_shape, dtype=data.dtype)
                    slices = tuple(slice(0, min(data.shape[i], target_shape[i])) 
                                   for i in range(len(target_shape)))
                    aligned[slices] = data[slices]
                    aligned_datasets.append(aligned)
                    aligned_dict[name] = aligned
                    print(f"    Cropped/padded shape: {aligned.shape}")
    else:
        # If no target shape, use original datasets
        aligned_datasets = list(datasets.values())
        aligned_dict = datasets
    
    # Print shape information for debugging
    print("\nFinal shapes:")
    for name, data in aligned_dict.items():
        print(f"  {name}: {data.shape}")
    
    # Save aligned data if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, data in aligned_dict.items():
            output_path = os.path.join(args.output_dir, f"{name}.tif")
            tifffile.imwrite(output_path, data)
            print(f"Saved aligned data to {output_path}")
    
    # Visualize
    print("\nStarting Napari visualization...")
    viewer = visualize_aligned_datasets(aligned_dict, names, args.voxel_size)
    napari.run()

if __name__ == "__main__":
    main()