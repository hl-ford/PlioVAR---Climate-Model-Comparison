import numpy as np
import numpy.ma as ma

def weighted_knn_interpolate(masked_array, target_coord, depth_values, lat_values, lon_values, k=8):
    """
    Perform weighted k-nearest neighbor interpolation on a 3D masked array.
    
    Parameters:
    -----------
    masked_array : numpy.ma.MaskedArray
        3D masked array with shape (depth, lat, lon)
    target_coord : tuple
        (depth, lat, lon) coordinate to interpolate at
    depth_values : numpy.ndarray
        Actual depth values corresponding to the depth dimension
    lat_values : numpy.ndarray
        Actual latitude values corresponding to the lat dimension
    lon_values : numpy.ndarray
        Actual longitude values corresponding to the lon dimension
    k : int
        Number of nearest neighbors to use (default: 8)
        
    Returns:
    --------
    float
        Interpolated value at the target coordinate, or masked value if interpolation
        is not possible (e.g., all surrounding points are masked)
    """
    # Extract target coordinates
    target_depth, target_lat, target_lon = target_coord
    
    # Check if target is within bounds
    if target_depth < depth_values.min() or target_depth > depth_values.max():
        return ma.masked
    
    if lat_values.ndim == 1:
        if (target_lat < lat_values.min() or target_lat > lat_values.max() or
            target_lon < lon_values.min() or target_lon > lon_values.max()):
            return ma.masked
    else:
        # For 2D lat/lon, check bounds after flattening
        if (target_lat < lat_values.min() or target_lat > lat_values.max() or
            target_lon < lon_values.min() or target_lon > lon_values.max()):
            return ma.masked
    
    # Initialize arrays for distances and data
    total_points = masked_array.size
    chunk_size = min(100000, total_points)  # Adjust chunk size based on available memory
    n_chunks = (total_points + chunk_size - 1) // chunk_size
    
    # Arrays to store k nearest neighbors
    k_distances = np.full(k, np.inf)
    k_values = np.zeros(k)
    
    # Process data in chunks
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_points)
        chunk_size_actual = end_idx - start_idx
        
        # Get indices for this chunk
        flat_indices = np.arange(start_idx, end_idx)
        if lat_values.ndim == 1:
            # For 1D coordinates
            d_indices, lat_indices, lon_indices = np.unravel_index(
                flat_indices, masked_array.shape)
            chunk_lats = lat_values[lat_indices]
            chunk_lons = lon_values[lon_indices]
        else:
            # For 2D coordinates
            d_indices, lat_indices, lon_indices = np.unravel_index(
                flat_indices, masked_array.shape)
            chunk_lats = lat_values[lat_indices, lon_indices]
            chunk_lons = lon_values[lat_indices, lon_indices]
        
        depth_scale = 1.0  # Adjust based on your data scale
        lat_lon_scale = 111000.0  # Approx meters per degree

        chunk_distances = np.sqrt(
            (depth_scale * (depth_values[d_indices] - target_depth))**2 +
            (lat_lon_scale * (chunk_lats - target_lat))**2 +
            (lat_lon_scale * np.cos(np.radians((chunk_lats + target_lat)/2)) * 
            (chunk_lons - target_lon))**2
        )
                
        # Get values and masks for this chunk
        chunk_values = masked_array.ravel()[start_idx:end_idx]
        chunk_masks = masked_array.mask.ravel()[start_idx:end_idx] if hasattr(masked_array, 'mask') else np.zeros(chunk_size_actual, dtype=bool)
        
        # Filter out masked points
        valid_indices = ~chunk_masks
        if np.any(valid_indices):
            valid_distances = chunk_distances[valid_indices]
            valid_values = chunk_values[valid_indices]
            
            # Update k nearest neighbors
            all_distances = np.concatenate([k_distances, valid_distances])
            all_values = np.concatenate([k_values, valid_values])
            
            # Find indices of k smallest distances
            smallest_k_indices = np.argpartition(all_distances, k)[:k]
            
            # Update arrays
            k_distances = all_distances[smallest_k_indices]
            k_values = all_values[smallest_k_indices]
    
    # Check if we found any valid points
    if np.all(np.isinf(k_distances)):
        return ma.masked
    
    # Calculate weights
    weights = 1 / (k_distances + 1e-10)
    weights = weights / np.sum(weights)
    
    # Calculate weighted average
    result = np.sum(weights * k_values)
    
    return result

def weighted_knn_interpolate_multiple(masked_array, target_coords, depth_values, lat_values, lon_values, k=8):
    """
    Perform weighted k-nearest neighbor interpolation for multiple target coordinates.
    
    Parameters:
    -----------
    masked_array : numpy.ma.MaskedArray
        3D masked array with shape (depth, lat, lon)
    target_coords : list or numpy.ndarray
        List of (depth, lat, lon) coordinates to interpolate at
    depth_values : numpy.ndarray
        Actual depth values corresponding to the depth dimension
    lat_values : numpy.ndarray
        Actual latitude values corresponding to the lat dimension
    lon_values : numpy.ndarray
        Actual longitude values corresponding to the lon dimension
    k : int
        Number of nearest neighbors to use (default: 8)
        
    Returns:
    --------
    numpy.ma.MaskedArray
        Array of interpolated values at the target coordinates
    """
    # Convert target_coords to numpy array if it's not already
    target_coords = np.asarray(target_coords)
    
    # Initialize results array
    results = np.zeros(len(target_coords))
    mask = np.zeros(len(target_coords), dtype=bool)
    
    # Loop through each target coordinate
    for i, target_coord in enumerate(target_coords):
        result = weighted_knn_interpolate(masked_array, target_coord, depth_values, lat_values, lon_values, k)
        
        # Check if result is masked
        if ma.is_masked(result):
            mask[i] = True
        else:
            results[i] = result
    
    # Create masked array for results
    return ma.array(results, mask=mask)

# Example usage and testing
if __name__ == "__main__":
    # Create a sample 3D masked array
    shape = (5, 4, 3)  # (depth, lat, lon)
    data = np.arange(60).reshape(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[2, 1, 1] = True  # Mask one point
    masked_data = ma.array(data, mask=mask)
    
    # Create coordinate arrays
    depth_values = np.array([0, 100, 200, 300, 400])  # meters
    lat_values = np.array([30, 32, 34, 36])  # degrees
    lon_values = np.array([-125, -120, -5])  # degrees
    
    # Test interpolation at a point
    target = (150, 33, -122.5)
    result = weighted_knn_interpolate(masked_data, target, depth_values, lat_values, lon_values)
    print(f"Interpolated value at {target}: {result}")
    
    # Test multiple points interpolation
    targets = [
        (150, 33, -122.5),
        (250, 33, -122.5),
        (350, 35, -117.5)
    ]
    results_multiple = weighted_knn_interpolate_multiple(masked_data, targets, depth_values, lat_values, lon_values)
    print(f"Interpolated values at multiple points: {results_multiple}")