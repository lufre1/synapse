import elf.parallel as parallel
import numpy as np
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, label
import skimage.segmentation as seg
import skimage.filters as filters
from skimage.draw import polygon
import skimage.morphology as morph
import scipy.ndimage as ndi
from skimage.morphology import binary_closing, ball, binary_dilation, convex_hull_image


def segment_mitos_from_labels_gemini(outer: np.ndarray, inner: np.ndarray):
    """
    Constructs a mask covering the volume enclosed by 'outer', excluding 'inner' (3D).

    Args:
        outer (np.ndarray): 3D binary mask representing the outer boundary.
        inner (np.ndarray): 3D binary mask representing the inner boundary.

    Returns:
        dict: A dictionary containing the final mask under the key "labels/mito".
    """
    outer = morph.dilation(_postprocess_seg_3d(outer), footprint=np.ones([5, 5, 5]))
    outer = morph.dilation(_postprocess_seg_3d(outer), footprint=np.ones([5, 5, 5]))
    outer = broaden_and_close_boundaries(outer, closing_radius=5)
    outer = morph.binary_closing(outer, footprint=np.ones([5, 5, 5]))
    if outer.ndim != 3:
        raise ValueError("Outer mask must be a 3D array.")

    # 1. Find bounding box of the outer mask
    z_coords, y_coords, x_coords = np.where(outer)

    if z_coords.size == 0 or y_coords.size == 0 or x_coords.size == 0:
        # Handle the case where the outer mask is empty
        return {"labels/mito": np.zeros_like(outer, dtype=bool)}

    min_z, max_z = np.min(z_coords), np.max(z_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)

    # 2. Crop the volume to the bounding box
    cropped_outer = outer[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    # 3. Process each slice (z-axis)
    cropped_closed_outer = np.zeros_like(cropped_outer)
    for z in range(cropped_outer.shape[0]):
        cropped_closed_outer[z] = convex_hull_image(cropped_outer[z])

    # 4. Reconstruct the full-size closed_outer
    closed_outer = np.zeros_like(outer)
    closed_outer[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = cropped_closed_outer

    closed_outer = binary_closing(closed_outer)

    # 5. Fill the Enclosed Volume
    filled_mask = ndi.binary_fill_holes(closed_outer)

    # 6. Remove Inner Volume
    dilated_inner = binary_dilation(inner)
    final_mask = np.logical_or(filled_mask, dilated_inner)

    return {"labels/mito": final_mask}


def broaden_and_close_boundaries(outer_boundary: np.ndarray, iterations: int = 3, closing_radius: int = 2) -> np.ndarray:
    """
    Broadens and closes outer boundaries in 3D volumetric data.

    Parameters:
        outer_boundary (np.ndarray): 3D binary array where 1 represents the boundary.
        iterations (int): Number of dilation iterations to broaden the boundary.
        closing_radius (int): Radius for morphological closing.

    Returns:
        np.ndarray: Processed 3D array with broadened and closed boundaries.
    """
    # Step 1: Dilate the boundary to broaden it
    dilated = ndi.binary_dilation(outer_boundary, structure=ball(1), iterations=iterations)

    # Step 2: Apply binary closing to connect gaps
    closed = binary_closing(dilated, ball(closing_radius))

    # Step 3: Fill holes inside mitochondria
    filled = ndi.binary_fill_holes(closed)

    return filled.astype(np.uint8)  # Convert to uint8 (0 and 1)


def segment_mitos_morphology(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """
    Segment mitochondria using morphological operations.
    
    Parameters:
    - outer: 3D array representing the outer boundary.
    - inner: 3D array representing the inner boundary (seed).
    
    Returns:
    - A 3D binary mask with segmented mitochondria.
    """
    # Initialize the segmentation result
    segmented = np.zeros_like(outer, dtype=np.uint8)

    # Iterate through slices along the z-axis
    for z in range(outer.shape[0]):
        inner_slice = inner[z]
        outer_slice = outer[z]

        # Skip if no seed is present in this slice
        if np.count_nonzero(inner_slice) == 0:
            continue

        # Start with the inner boundary as a seed
        mask = np.copy(inner_slice)

        # Iteratively dilate the seed until it reaches the outer boundary
        while True:
            expanded = morph.binary_dilation(mask, morph.disk(2))  # Grow the region
            if np.array_equal(expanded, mask):  # Stop if no more growth
                break
            mask = np.where(outer_slice, 0, expanded)  # Prevent growth beyond the outer boundary

        # Store the segmented region
        segmented[z] = mask

    return segmented


def segment_mitos_with_snake(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """
    Segment mitochondria using a 3D active contour (snake) approach.
    
    Parameters:
    - outer: 3D array representing the outer boundary.
    - inner: 3D array representing the inner boundary (used as seed).
    
    Returns:
    - A 3D binary mask with segmented mitochondria.
    """
    # Initialize result mask
    segmented = np.zeros_like(outer, dtype=np.uint8)

    # Iterate through slices along the z-axis
    for z in range(outer.shape[0]):
        outer_slice = outer[z]
        inner_slice = inner[z]

        if np.count_nonzero(inner_slice) == 0:
            continue  # Skip slices without seeds
        
        # Extract seed points from inner boundary
        seed_points = np.column_stack(np.nonzero(inner_slice))
        
        # Create an initial mask using outer boundary
        outer_points = np.column_stack(np.nonzero(outer_slice))
        
        # Convert outer boundary to a polygon and fill it
        if len(outer_points) > 3:
            rr, cc = polygon(outer_points[:, 0], outer_points[:, 1], outer_slice.shape)
            mask = np.zeros_like(outer_slice, dtype=np.uint8)
            mask[rr, cc] = 1
        else:
            continue  # Skip if not enough outer boundary points

        # Apply watershed segmentation using the inner boundary as seeds
        distance_map = filters.sobel(mask)
        markers = np.zeros_like(mask)
        markers[inner_slice > 0] = 1
        markers[outer_slice > 0] = 2
        segmentation = seg.watershed(distance_map, markers) == 1

        # Store the segmented result
        segmented[z] = segmentation

    return segmented


def segment_mitos_from_labels(
    outer: np.ndarray,
    inner: np.ndarray,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
) -> dict:
    """
    Segments mitochondria using the watershed algorithm,
    using inner boundaries as seeds.

    Args:
        outer (np.ndarray): Binary mask of the outer boundaries.
        inner (np.ndarray): Binary mask of the inner boundaries (seeds).

    Returns:
        np.ndarray: Labeled segmentation mask of mitochondria.
    """
    # Compute distance transform from inner boundaries
    # distance = distance_transform_edt(inner)
    # outer = morph.dilation(_postprocess_seg_3d(outer), footprint=np.ones([5, 5, 5]))
    outer = morph.dilation(_postprocess_seg_3d(outer, iterations_3d=15), footprint=np.ones([5, 5, 5]))
    outer = broaden_and_close_boundaries(outer, closing_radius=5)
    outer = morph.binary_closing(outer, footprint=np.ones([5, 5, 5]))
    dist = parallel.distance_transform(outer,
                                       verbose=True,
                                       halo=halo,
                                       block_shape=block_shape)
    # hmap = filters.gaussian(dist)
    hmap = ((dist.max() - dist) / dist.max())

    # Label connected inner boundary components as markers
    seeds = morph.dilation(_postprocess_seg_3d(inner.copy()), footprint=np.ones([5, 5, 5]))
    seg = np.zeros_like(seeds)
    # Compute watershed segmentation using distance as input
    seg = parallel.seeded_watershed(hmap,
                                    seeds,
                                    out=seg,
                                    verbose=True,
                                    block_shape=block_shape,
                                    halo=halo
                                    )  # mask=(outer | inner)

    return {
        "labels/mitos": seg,
        "dist": dist,
        "hmap": hmap,
        "seeds": seeds,
        "new_outer": outer
    }


def segment_from_pred(pred,
                      block_shape=(128, 256, 256),
                      halo=(48, 48, 48),
                      seed_distance=6 * 1,
                      boundary_threshold=0.25,
                      min_size=50000*8,
                      area_threshold=1000 * 5,
                      dist=None
                      ) -> dict:
    foreground, boundaries = pred
    # # #boundaries = binary_erosion(boundaries < boundary_threshold, structure=np.ones((1, 3, 3)))
    if dist is None:
        dist = parallel.distance_transform(boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    # # data["pred_dist_without_fore"] = parallel.distance_transform((boundaries) < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    hmap = ((dist.max() - dist) / dist.max())

    hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()

    # # hmap = hmap.clip(min=0)
    seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=True)
    # # #seeds = binary_fill_holes(seeds)

    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=True, halo=halo,
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    seg_data = {
        "seg": seg,
        "seeds": seeds,
        "dist": dist,
        "hmap": hmap
    }
    return seg_data
