"""Area/volume-weighted summary statistics for the Atlantic below 200 m."""

import numpy as np
import numpy.ma as ma

from .gridding import connect_atlantic
from .gridding import resolve_latlon_mesh
from .masks import apply_atlantic_mask_3d


def _wrapped_gradient_deg(values, axis):
    '''Gradient of a coordinate in degrees, robust to the 360 -> 0 longitude jump.'''
    grad = np.gradient(np.asarray(values, dtype=float), axis=axis)
    return (grad + 90.0) % 180.0 - 90.0


def cell_areas_km2(lat, lon, radius=6371.0):
    '''
    Input: lat and lon (1D vectors or 2D meshes, regular or curvilinear).
    Output: 2D (lat, lon) approximate horizontal cell areas in km^2, from cell-centre spacing.
    '''
    lat2d, lon2d = resolve_latlon_mesh(lat, lon)
    coslat = np.cos(np.deg2rad(np.asarray(lat2d, dtype=float)))
    dlat_j = np.deg2rad(_wrapped_gradient_deg(lat2d, axis=0))
    dlat_i = np.deg2rad(_wrapped_gradient_deg(lat2d, axis=1))
    dlon_j = np.deg2rad(_wrapped_gradient_deg(lon2d, axis=0))
    dlon_i = np.deg2rad(_wrapped_gradient_deg(lon2d, axis=1))
    # Cross product of the local (east, north) steps along each grid axis
    return np.abs(dlon_i * coslat * dlat_j - dlon_j * coslat * dlat_i) * radius**2


def layer_thickness_m(depth, depth_is_top=False):
    '''
    Input: 1D depth levels in metres (cell centres such as z_t, or cell tops such as z_w).
    Output: 1D layer thicknesses in metres.
    '''
    depth = np.asarray(depth, dtype=float)
    if depth_is_top:
        edges = np.concatenate((depth, [depth[-1] + (depth[-1] - depth[-2])]))
    else:
        edges = np.concatenate(([0.0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1] + 0.5 * (depth[-1] - depth[-2])]))
    return np.diff(edges)


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, cumulative, values)


def atlantic_summary_stats(field, lat, lon, depth, min_depth=200.0, depth_is_top=False):
    '''
    Input: 3D (depth, lat, lon) field on its native grid, its lat/lon (1D or 2D) and 1D depth in metres.
    Output: dict of volume-weighted statistics inside the Atlantic polygon (same one as the
    lat-depth figures, via apply_atlantic_mask_3d), excluding levels shallower than min_depth.
    '''
    lat2d, lon2d = resolve_latlon_mesh(lat, lon)
    lon2d = connect_atlantic(lon2d)
    masked = apply_atlantic_mask_3d(ma.masked_invalid(field), lat2d, lon2d, depth)

    keep = np.asarray(depth) >= min_depth
    values = masked[keep]
    weights = (layer_thickness_m(depth, depth_is_top)[keep, None, None]
               * cell_areas_km2(lat2d, lon2d)[None, :, :] / 1000.0)  # km^3
    weights = np.broadcast_to(weights, values.shape)

    valid = ~ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
    x = values.data[valid].astype(float)
    w = weights[valid]

    mean = np.average(x, weights=w)
    p10, median, p90 = _weighted_quantile(x, w, [0.1, 0.5, 0.9])
    return {
        "n_cells": x.size,
        "volume_km3": w.sum(),
        "mean": mean,
        "std": np.sqrt(np.average((x - mean) ** 2, weights=w)),
        "median": median,
        "p10": p10,
        "p90": p90,
        "min": x.min(),
        "max": x.max(),
    }
