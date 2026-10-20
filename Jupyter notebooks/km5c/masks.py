"""Ocean-basin polygons and masking helpers."""

from matplotlib.path import Path
import numpy as np
import numpy.ma as ma

from .gridding import regrid_mask_to_data


atlantic_polygon = Path([(lon, lat) for lon, lat in zip([20, 20, -3, -3, 20, 20, -70, -60, -77, -79.3, -81, -84.4, -90, -90], [70, 37, 37, 36, 36, -50, -50, 0, 8, 9.3, 8.3, 10.6, 15, 70])])
#atlantic_polygon = Path([(lon, lat) for lon, lat in zip([20, 20, -70, -60, -77, -79.3, -81, -84.4, -90, -90], [70, -50, -50, 0, 8, 9.3, 8.3, 10.6, 15, 70])])
southern_polygon = Path([
    (-180, -50),  # Start from the westernmost point at -50 latitude
    (180, -50),   # Move eastward to the easternmost point at -50 latitude
    (180, -70),   # Go down to -70 latitude at the easternmost point
    (-180, -70),  # Move westward to the westernmost point at -70 latitude
    (-180, -50)   # Close the polygon by returning to the starting point
])


def apply_mask(array, data_lat, data_lon, mask=None, mask_lat=None, mask_lon=None):
    arr = np.ma.array(array, copy=False)

    if mask is None:
        return arr

    if mask_lat is None or mask_lon is None:
        raise ValueError("mask_lat and mask_lon are required when mask is provided.")

    mask_on_data = regrid_mask_to_data(mask, mask_lat, mask_lon, data_lat, data_lon, method="nearest")

    if arr.ndim == 2:
        if arr.shape != mask_on_data.shape:
            raise ValueError(f"2D shape mismatch after remap: array={arr.shape}, mask={mask_on_data.shape}")
        return np.ma.masked_where(~mask_on_data, arr)

    if arr.ndim == 3:
        if arr.shape[-2:] != mask_on_data.shape:
            raise ValueError(f"3D shape mismatch after remap: array={arr.shape}, mask={mask_on_data.shape}")
        out = np.ma.array(arr, copy=True)
        for k in range(out.shape[0]):
            out[k] = np.ma.masked_where(~mask_on_data, out[k])
        return out

    raise ValueError("array must be 2D or 3D")


# Examples:
# Plio_d18O_masked = apply_mask(Plio_d18O, mask="icesm13")
# preind_d18O_masked = apply_mask(preind_d18O, mask="hadcm3")
# temp_masked = apply_mask(Plio_temp, mask=icesm13_basin_mask)  # direct custom mask
def apply_atlantic_mask_2d(array, lat, lon):
    '''
    Input: 2D array (benthic), latitude, and longitude mesh arrays.
    Output: Masked array, with points outside the Atlantic polygon masked.
    '''
    # Define Atlantic polygon coordinates
    atlantic_polygon = Path([
        (-70, 70), (20, 70), (20, 50), (0, 50), (0, 40), (-5, 40), 
        (-5, 20), (20, 20), (20, -90), (-70, -90), 
        (-70, 12.8396), (-90, 12.8396), (-90, 50), (-70, 50)
    ])
    
    # Create a mask based on whether points are inside the polygon
    points = np.column_stack((lon.flatten(), lat.flatten()))
    geographical_mask = atlantic_polygon.contains_points(points).reshape(lat.shape)
    
    if ma.isMaskedArray(array):
        # If the input array is already masked, combine the existing mask with the geographical mask
        original_mask = ma.getmask(array)
        combined_mask = np.logical_or(original_mask, ~geographical_mask)
        masked_array = ma.masked_where(combined_mask, array)
    else:
        # Apply the geographical mask directly to the array
        masked_array = ma.masked_where(~geographical_mask, array)
    
    return masked_array


def apply_atlantic_mask_3d(array, lat, lon, depth, mask = "atlantic_polygon"):
    '''
    Input: 3D array, Atlantic-connected latitude, longitude mesh arrays and depth 1D arrays.
    Output: Masked array, with points outside the Atlantic polygon masked.
    '''

    # Define Atlantic polygon coordinates
    atlantic_polygon = Path([
        (-70, 70), (20, 70), (20, 50), (0, 50), (0, 40), (-5, 40), 
        (-5, 20), (20, 20), (20, -90), (-70, -90), 
        (-70, 12.8396), (-90, 12.8396), (-90, 50), (-70, 50)
    ])
    depth_values = np.arange(len(depth))
    masked_array = []

    for depth in depth_values:
        array_slice = array[depth, :, :] 
        points = np.column_stack((lon.flatten(), lat.flatten()))
        geographical_mask = atlantic_polygon.contains_points(points).reshape(lat.shape)
        if ma.isMaskedArray(array):
            original_mask = ma.getmask(array_slice)
            combined_mask = np.logical_or(original_mask, ~geographical_mask)
            masked_slice = ma.masked_where(combined_mask, array_slice)
        else:
            # Apply the geographical mask directly to the array
            masked_slice = ma.masked_where(~geographical_mask, array_slice)
            
        masked_array.append(masked_slice)
    full_masked_array = ma.stack(masked_array, axis=0)
    return full_masked_array
