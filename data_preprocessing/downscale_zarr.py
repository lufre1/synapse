import zarr
import numpy as np
from skimage.transform import rescale
from tqdm import tqdm

def downscale_zarr_dataset(zarr_path, input_key="0", output_key="1", scale_factor=0.5, z_block_size=64):
    print(f"Opening Zarr store at: {zarr_path}")
    root = zarr.open(zarr_path, mode='a')
    
    if input_key not in root:
        raise KeyError(f"Dataset '{input_key}' not found in the Zarr store.")
        
    data_in = root[input_key]
    old_shape = data_in.shape
    
    # Calculate new shape (e.g., dividing Z, Y, X by 2)
    new_shape = tuple(int(s * scale_factor) for s in old_shape)
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
        rescaled_block = rescale(
            block, 
            scale=scale_factor, 
            preserve_range=True, 
            anti_aliasing=True,
            channel_axis=None # Explicitly state this is a spatial 3D volume, not channels
        )
        
        # Convert back to the original datatype (rescale outputs floats by default)
        rescaled_block = rescaled_block.astype(data_in.dtype)
        
        # Calculate the corresponding Z-indices in the downscaled dataset
        z_out_start = int(z * scale_factor)
        z_out_end_expected = z_out_start + rescaled_block.shape[0]
        
        # Clamp the end index so we don't exceed the new total shape
        z_out_end = min(z_out_end_expected, new_shape[0])
        
        # Calculate how many slices we are actually allowed to write
        valid_z_len = z_out_end - z_out_start
        
        # Write the rescaled block to disk, slicing off any extra frame at the end
        data_out[z_out_start:z_out_end] = rescaled_block[:valid_z_len]

    print(f"Successfully created dataset '{output_key}'!")

if __name__ == "__main__":
    ZARR_FILE = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_zarr/4007/images/ome-zarr/raw.ome.zarr"
    
    # Using z_block_size=64 means we load 64 slices at a time. 
    # If your images are massive in X and Y (e.g., >10,000 pixels) and you still run out of RAM, 
    # lower this number to 32 or 16.
    downscale_zarr_dataset(ZARR_FILE, z_block_size=64)