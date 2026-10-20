"""Regridding, interpolation and nearest-neighbour helpers."""

from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from sklearn.impute import KNNImputer
import numpy as np
import numpy.ma as ma


def _regrid_curvilinear_common_mask(source_field, source_lat, source_lon,
                                   target_field, target_lat, target_lon):
    """Interpolate a depth/lat/lon field to a regular grid with a shared mask.

    Periodic longitude copies let the Delaunay triangulation cross the dateline.
    Each target value uses barycentric interpolation from its source triangle;
    it is retained only when every source vertex with nonzero weight and the
    target-grid value are valid. This conservative rule avoids filling ocean
    cells across masked source values. The triangulation and target weights are
    built once and reused at every depth.
    """
    source = ma.asarray(source_field)
    target = ma.asarray(target_field)
    source_lat = ma.asarray(source_lat)
    source_lon = ma.asarray(source_lon)
    target_lat = ma.asarray(target_lat)
    target_lon = ma.asarray(target_lon)

    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("Source and target fields must have depth, latitude, longitude dimensions.")
    if source_lat.shape != source_lon.shape or source.shape[1:] != source_lat.shape:
        raise ValueError("Source coordinates must match the source field's horizontal grid.")
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("Target latitude and longitude coordinates must be one-dimensional.")
    if source.shape[0] != target.shape[0]:
        raise ValueError("Source and target fields must have the same number of depth levels.")
    if target.shape[1:] != (target_lat.size, target_lon.size):
        raise ValueError("Target field shape must match its regular latitude/longitude coordinates.")

    source_lat_values = np.asarray(source_lat, dtype=float)
    source_lon_values = np.asarray(source_lon, dtype=float)
    source_coord_valid = (
        ~ma.getmaskarray(source_lat)
        & ~ma.getmaskarray(source_lon)
        & np.isfinite(source_lat_values)
        & np.isfinite(source_lon_values)
        & (np.abs(source_lat_values) <= 90)
    )
    source_indices = np.flatnonzero(source_coord_valid.ravel())
    if source_indices.size < 3:
        raise ValueError("At least three finite source coordinates are required for regridding.")

    base_lon = np.mod(source_lon_values.ravel()[source_indices], 360.0)
    base_lat = source_lat_values.ravel()[source_indices]
    base_points = np.column_stack((base_lon, base_lat))
    points = np.concatenate((
        base_points + np.array([-360.0, 0.0]),
        base_points,
        base_points + np.array([360.0, 0.0]),
    ))
    point_source_indices = np.tile(source_indices, 3)

    # Coincident coordinates can occur at a periodic seam. Collapse them for
    # Qhull while retaining all original cells for conservative validity checks.
    unique_points, point_group = np.unique(points, axis=0, return_inverse=True)
    group_sizes = np.bincount(point_group, minlength=unique_points.shape[0])
    triangulation = Delaunay(unique_points)

    target_lat_values = np.asarray(target_lat, dtype=float)
    target_lon_values = np.asarray(target_lon, dtype=float)
    if (
        ma.getmaskarray(target_lat).any()
        or ma.getmaskarray(target_lon).any()
        or not np.all(np.isfinite(target_lat_values))
        or not np.all(np.isfinite(target_lon_values))
    ):
        raise ValueError("Target latitude and longitude coordinates must be unmasked and finite.")
    target_lon_grid, target_lat_grid = np.meshgrid(
        np.mod(target_lon_values, 360.0), target_lat_values
    )
    target_points = np.column_stack((target_lon_grid.ravel(), target_lat_grid.ravel()))
    target_simplex = triangulation.find_simplex(target_points)
    inside = target_simplex >= 0
    inside_positions = np.flatnonzero(inside)
    simplices = target_simplex[inside]
    triangle_vertices = triangulation.simplices[simplices]
    offsets = target_points[inside] - triangulation.transform[simplices, 2]
    first_weights = np.einsum(
        "nij,nj->ni", triangulation.transform[simplices, :2], offsets
    )
    triangle_weights = np.column_stack((first_weights, 1.0 - first_weights.sum(axis=1)))
    # Remove tiny roundoff at triangle edges and renormalize the barycentric weights.
    triangle_weights = np.clip(triangle_weights, 0.0, None)
    triangle_weights /= triangle_weights.sum(axis=1, keepdims=True)
    contributing = triangle_weights > 1e-12

    source_values = np.asarray(source, dtype=float).reshape(source.shape[0], -1)
    target_values = np.asarray(target, dtype=float)
    source_masks = ma.getmaskarray(source).reshape(source.shape[0], -1)
    target_masks = ma.getmaskarray(target)
    regridded = np.full(target.shape, np.nan, dtype=float)
    common_mask = np.ones(target.shape, dtype=bool)

    for level in range(source.shape[0]):
        source_valid = ~source_masks[level] & np.isfinite(source_values[level])
        point_valid = source_valid[point_source_indices]
        valid_count = np.bincount(
            point_group, weights=point_valid.astype(float), minlength=unique_points.shape[0]
        )
        vertex_valid = valid_count == group_sizes

        point_values = source_values[level, point_source_indices]
        point_values = np.where(point_valid, point_values, 0.0)
        value_sum = np.bincount(
            point_group, weights=point_values, minlength=unique_points.shape[0]
        )
        vertex_values = value_sum / group_sizes

        selected_valid = vertex_valid[triangle_vertices]
        valid_contributors = np.all(~contributing | selected_valid, axis=1)
        selected_values = vertex_values[triangle_vertices]
        interpolated = np.sum(
            np.where(contributing, selected_values, 0.0) * triangle_weights, axis=1
        )

        target_level = target_values[level].ravel()
        target_valid = ~target_masks[level].ravel() & np.isfinite(target_level)
        valid_target = np.zeros(target_level.shape, dtype=bool)
        valid_target[inside_positions] = valid_contributors & target_valid[inside_positions]
        level_output = regridded[level].ravel()
        level_output[inside_positions] = interpolated
        common_mask[level] = ~valid_target.reshape(target.shape[1:])

    # Both fields receive the exact same depth-dependent intersection mask.
    isotope_on_target = ma.array(regridded, mask=common_mask, copy=False)
    temperature_on_target = ma.array(target_values, mask=common_mask, copy=False)
    return isotope_on_target, temperature_on_target


def create_mesh(lat, lon):
    '''
    Input: 1D lat and lon array
    Returns: 2D lat and lon mesh
    '''
    lonmesh, latmesh = np.meshgrid(lon, lat)
    return latmesh, lonmesh


def connect_atlantic(lon):
    '''Input: longitude mesh from 0-360, returns longitude mesh from -180 to 180'''
    return (lon + 180) % 360 - 180


def get_bottom(array):
    '''
    Input: (depth, lat, lon) array (can be a masked array)
    Output: masked array with benthic (last unmasked) points of (lat, lon) array
    '''
    # Check if the input array has the correct dimensions
    if len(array.shape) != 3:
        raise ValueError("Input array must have dimensions (depth, lat, lon)")

    depth, lat, lon = array.shape
    bottom_points = np.zeros((lat, lon))  # Initialize an array for bottom points
    mask = np.ones((lat, lon), dtype=bool)  # Initialize a mask array (True means masked)

    # Iterate over each (lat, lon) point
    for i in range(lat):
        for j in range(lon):
            # Get the values along the depth for this (lat, lon) point
            depth_column = array[:, i, j]
            
            # Find the last unmasked value (searching from the bottom)
            for d in range(depth - 1, -1, -1):  # Start from the bottom and move upwards
                if not np.ma.is_masked(depth_column[d]):  # Check if the value is unmasked
                    bottom_points[i, j] = depth_column[d]
                    mask[i, j] = False  # Unmask this point
                    break  # Stop at the first unmasked value found

    # Return a masked array where masked points remain
    return ma.masked_array(bottom_points, mask=mask)


def interpolate(lat, lon, benthic_array, newlat = 1000, newlon = 1000, grid = True):
    '''Input: latitude and longitude mesh, original benthic array
    Returns: interpolated interpolated array '''
    if not isinstance(grid, bool):
        raise TypeError(f"Expected 'grid' to be a boolean, but got {type(grid).__name__}")
    if grid == False:
        fine_lat = np.linspace(lat.min(), lat.max(), newlat)
        fine_lon = np.linspace(lon.min(), lon.max(), newlon)
    if grid == True:
        fine_lat = np.linspace(-90, 90, newlat)
        fine_lon = np.linspace(-180, 180, newlon)
        
    
    fine_lon_mesh, fine_lat_mesh = np.meshgrid(fine_lon, fine_lat)
    interpolated_array = griddata((lat.flatten(), 
                                   lon.flatten()), 
                                  benthic_array.flatten(), 
                                  (fine_lat_mesh, 
                                   fine_lon_mesh), 
                                  method='linear')
    if np.ma.isMaskedArray(benthic_array):
        mask = ma.getmaskarray(benthic_array)  # always full bool array shape
        fine_mask = griddata(
            (lat.flatten(), lon.flatten()),
            mask.astype(float).flatten(),
            (fine_lat_mesh, fine_lon_mesh),
            method="nearest"
        )

        fine_mask = np.asarray(fine_mask)
        fine_mask = np.nan_to_num(fine_mask, nan=1.0) >= 0.5  # force bool mask

        interpolated_array[fine_mask] = np.nan
        interpolated_array = ma.masked_where(fine_mask, interpolated_array)

    return interpolated_array, fine_lat_mesh, fine_lon_mesh


def resolve_latlon_mesh(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        return np.meshgrid(lat, lon, indexing="ij")
    if lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(f"lat/lon mesh shape mismatch: {lat.shape} vs {lon.shape}")
        return lat, lon
    raise ValueError(f"Unsupported lat/lon dims: lat={lat.ndim}, lon={lon.ndim}")


def regrid_mask_to_data(mask2d, mask_lat, mask_lon, data_lat, data_lon, method="nearest"):
    mask2d = np.asarray(mask2d, dtype=bool)

    mlat2d, mlon2d = resolve_latlon_mesh(mask_lat, mask_lon)
    dlat2d, dlon2d = resolve_latlon_mesh(data_lat, data_lon)

    mlon2d = connect_atlantic(mlon2d)
    dlon2d = connect_atlantic(dlon2d)

    points = np.column_stack((mlat2d.ravel(), mlon2d.ravel()))
    values = mask2d.astype(float).ravel()  # True=keep, False=mask out

    out = griddata(points, values, (dlat2d, dlon2d), method=method, fill_value=0.0)
    return np.asarray(out >= 0.5, dtype=bool)


def compress_longitude(array):
    if array.ndim == 3:
        # Assuming the compression should happen along axis 2 for 3D arrays
        compressed_array = ma.mean(array, axis=2)
    elif array.ndim == 2:
        # Assuming the compression should happen along axis 1 for 2D arrays
        compressed_array = ma.mean(array, axis=1)
    else:
        raise ValueError(f"Array has an unsupported number of dimensions: {array.ndim}")
    
    return compressed_array


def compress_latitude(array):
    if array.ndim == 3:
        # Assuming the compression should happen along axis 2 for 3D arrays
        compressed_array = ma.mean(array, axis=2)
    elif array.ndim == 2:
        # Assuming the compression should happen along axis 1 for 2D arrays
        compressed_array = ma.mean(array, axis=0)
    else:
        raise ValueError(f"Array has an unsupported number of dimensions: {array.ndim}")
    
    return compressed_array


def KNN_implementation(array, axis1, axis2, axis1type="lat", axis2type="lon", neighbors=5):
    """
    Input:
      - array: 2D ndarray or masked array (values to impute)
      - axis1, axis2: same shape as array (2D meshes) describing coordinates
      - axis1type/axis2type: "lat", "lon", or "depth" (controls scaling)
    Output:
      - imputed 2D ndarray (same shape as array)
    """
    if array.shape != axis1.shape or array.shape != axis2.shape:
        raise ValueError(f"Shapes must match. array{array.shape}, axis1{axis1.shape}, axis2{axis2.shape}")
    # Convert input to a float array with NaNs where missing
    if ma.isMaskedArray(array):
        values = array.filled(np.nan).astype(float, copy=False)
    else:
        values = np.asarray(array, dtype=float)
    a1 = np.asarray(axis1, dtype=float)
    a2 = np.asarray(axis2, dtype=float)
    # Scale degrees -> km for lat/lon. If lon is involved, apply cos(lat) correction.
    km_per_deg = 111.0
    a1_scaled = a1.copy()
    a2_scaled = a2.copy()
    if axis1type in ("lat", "lon"):
        a1_scaled *= km_per_deg
    if axis2type in ("lat", "lon"):
        a2_scaled *= km_per_deg
    # If one axis is longitude, correct longitude distances by cos(latitude)
    # (Needs a latitude field; we assume the other axis is latitude if provided)
    lat_for_cos = None
    if axis1type == "lat":
        lat_for_cos = a1
    elif axis2type == "lat":
        lat_for_cos = a2
    if lat_for_cos is not None:
        coslat = np.cos(np.deg2rad(lat_for_cos))
        coslat = np.clip(coslat, 1e-6, None)  # avoid collapse at poles
        if axis1type == "lon":
            a1_scaled *= coslat
        if axis2type == "lon":
            a2_scaled *= coslat
    # Build [coord1, coord2, value] rows
    X = np.column_stack([a1_scaled.ravel(), a2_scaled.ravel(), values.ravel()])
    imputer = KNNImputer(n_neighbors=int(neighbors), weights="distance")
    X_imp = imputer.fit_transform(X)
    return X_imp[:, 2].reshape(array.shape)


def interpolate_to_common_grid(lat1, lon1, data1, lat2, lon2, data2):
    # Ensure lat/lon grids have compatible shapes
    if lat1.shape != lon1.shape:
        raise ValueError("Latitude and Longitude grids for dataset 1 do not have the same shape.")
    if lat2.shape != lon2.shape:
        raise ValueError("Latitude and Longitude grids for dataset 2 do not have the same shape.")
    
    # Define the common grid, choosing the intersection of the latitude ranges
    common_lat = np.linspace(max(np.min(lat1), np.min(lat2)), min(np.max(lat1), np.max(lat2)), 1000)
    common_lon = np.linspace(max(np.min(lon1), np.min(lon2)), min(np.max(lon1), np.max(lon2)), 1000)
    
    # Create a meshgrid for common lat/lon
    common_lat_grid, common_lon_grid = np.meshgrid(common_lat, common_lon)
    
    # Flatten the grids for interpolation
    points1 = np.array([lat1.flatten(), lon1.flatten()]).T
    points2 = np.array([lat2.flatten(), lon2.flatten()]).T
    
    # Check if data1 and data2 are flattening to the same size as their respective points
    if points1.shape[0] != data1.flatten().shape[0]:
        raise ValueError(f"Data1 has {data1.flatten().shape[0]} values, but points1 has {points1.shape[0]} coordinates.")
    if points2.shape[0] != data2.flatten().shape[0]:
        raise ValueError(f"Data2 has {data2.flatten().shape[0]} values, but points2 has {points2.shape[0]} coordinates.")
    
    # Interpolate data1 and data2 to the common grid
    data1_interpolated = griddata(points1, data1.flatten(), (common_lat_grid, common_lon_grid), method='linear', fill_value=np.nan)
    data2_interpolated = griddata(points2, data2.flatten(), (common_lat_grid, common_lon_grid), method='linear', fill_value=np.nan)
    
    # Combine the original masks of data1 and data2
    mask1_original = np.ma.getmaskarray(data1)  # Original mask for data1
    mask2_original = np.ma.getmaskarray(data2)  # Original mask for data2

    # Interpolate the masks as well
    mask1_interpolated = griddata(points1, mask1_original.flatten().astype(float), (common_lat_grid, common_lon_grid), method='nearest', fill_value=1)
    mask2_interpolated = griddata(points2, mask2_original.flatten().astype(float), (common_lat_grid, common_lon_grid), method='nearest', fill_value=1)

    # Create a combined mask: mask where either dataset has invalid data (mask == 1.0 means invalid)
    combined_mask = (mask1_interpolated == 1) | (mask2_interpolated == 1)

    # Apply the combined mask to both interpolated datasets
    data1_interpolated_masked = np.ma.masked_array(data1_interpolated, mask=combined_mask)
    data2_interpolated_masked = np.ma.masked_array(data2_interpolated, mask=combined_mask)

    return common_lat_grid, common_lon_grid, data1_interpolated_masked, data2_interpolated_masked


def interpolate_depth_profile(target_depths, source_depths, source_profile_masked):
    """
    Interpolates a 1D masked data profile to target depth coordinates.

    Args:
        target_depths (np.ndarray): The new 1D depth coordinates to interpolate onto.
        source_depths (np.ndarray): The original 1D depth coordinates of the profile.
        source_profile_masked (np.ma.MaskedArray): The 1D masked data profile.

    Returns:
        np.ma.MaskedArray: The interpolated 1D masked profile on target_depths.
    """
    # Get unmasked data and corresponding depths from the source profile
    if not isinstance(source_profile_masked, np.ma.MaskedArray):
        # If not a masked array, treat as unmasked
        source_profile_masked = np.ma.masked_array(source_profile_masked, mask=np.zeros_like(source_profile_masked, dtype=bool))
    
    valid_points_mask = ~source_profile_masked.mask
    
    # Check if there are enough valid (unmasked) points in the source profile to interpolate
    if np.sum(valid_points_mask) < 2:
        # Not enough data points to perform interpolation, return a fully masked array
        return np.ma.masked_all(target_depths.shape, dtype=source_profile_masked.dtype if hasattr(source_profile_masked, 'dtype') else float)

    valid_source_depths = source_depths[valid_points_mask]
    valid_source_data = source_profile_masked.data[valid_points_mask]

    # Create data interpolation function (linear interpolation)
    # Extrapolated points (outside original valid_source_depths range) will be NaN
    data_interp_func = interp1d(valid_source_depths, valid_source_data, kind='linear',
                                bounds_error=False, fill_value=np.nan)
    interpolated_data = data_interp_func(target_depths)

    # Interpolate the mask using nearest-neighbor
    # True (masked) is 1.0, False (unmasked) is 0.0
    # Points outside original source_depths range will be filled with 1.0 (masked)
    if len(source_depths) == 0 : # Should not happen if sum(valid_points_mask) >=2
         mask_interp_func = lambda x: np.ones_like(x, dtype=float) # all masked
    else:
        mask_interp_func = interp1d(source_depths, source_profile_masked.mask.astype(float),
                                kind='nearest', bounds_error=False, fill_value=1.0)
    
    interpolated_mask_float = mask_interp_func(target_depths)
    interpolated_mask_bool = (interpolated_mask_float >= 0.5) # Convert back to boolean mask

    # Combine masks: also mask where interpolated data became NaN (e.g., from extrapolation)
    final_mask = np.isnan(interpolated_data) | interpolated_mask_bool
    
    return np.ma.masked_array(interpolated_data, mask=final_mask)
