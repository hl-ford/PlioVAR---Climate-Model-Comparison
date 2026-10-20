"""Great-circle transects and projection of proxy sites onto them."""

from scipy.interpolate import griddata
import math
import numpy as np
import numpy.ma as ma
import pandas as pd

from .gridding import connect_atlantic
from .gridding import create_mesh


def great_circle_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c
    return distance


def wrap_longitude_180(lon_values):
    lon_values = np.asarray(lon_values, dtype=float)
    return ((lon_values + 180) % 360) - 180


def lon_abs_diff_deg(a, b):
    return np.abs(((a - b + 180) % 360) - 180)


def _wrap_lon_diff_deg(lon_values, lon_center):
    return (lon_values - lon_center + 180.0) % 360.0 - 180.0


def _densify_transect_line(line_lon, line_lat, spacing_km=25.0):
    line_lon = np.asarray(line_lon, dtype=float)
    line_lat = np.asarray(line_lat, dtype=float)

    bend_dist = [0.0]
    for i in range(1, len(line_lon)):
        bend_dist.append(
            bend_dist[-1] + great_circle_distance(
                line_lat[i - 1], line_lon[i - 1], line_lat[i], line_lon[i]
            )
        )
    bend_dist = np.asarray(bend_dist)

    s_lon = [line_lon[0]]
    s_lat = [line_lat[0]]
    s_dist = [0.0]

    for i in range(len(line_lon) - 1):
        lon0, lat0 = line_lon[i], line_lat[i]
        lon1, lat1 = line_lon[i + 1], line_lat[i + 1]
        seg_km = great_circle_distance(lat0, lon0, lat1, lon1)
        n = max(1, int(np.ceil(seg_km / spacing_km)))
        tvals = np.linspace(0, 1, n + 1)[1:]
        for t in tvals:
            lt = lon0 + t * (lon1 - lon0)
            la = lat0 + t * (lat1 - lat0)
            s_dist.append(s_dist[-1] + great_circle_distance(s_lat[-1], s_lon[-1], la, lt))
            s_lon.append(lt)
            s_lat.append(la)

    return np.asarray(s_lon), np.asarray(s_lat), np.asarray(s_dist), bend_dist


def extract_transect_section_lon_band_mean(
    array3d, lat, lon, depth, line_lon, line_lat,
    spacing_km=25.0, lon_half_width_deg=5.0,
    newlat=600, newlon=600, mesh=False, neighbors=8
):
    if array3d.ndim != 3:
        raise ValueError("array3d must be shape (depth, lat, lon)")

    if not mesh:
        latmesh, lonmesh = create_mesh(lat, lon)
    else:
        latmesh, lonmesh = lat, lon

    lonmesh = connect_atlantic(lonmesh)
    line_lon = np.asarray(line_lon, dtype=float)
    line_lat = np.asarray(line_lat, dtype=float)
    s_lon, s_lat, s_dist, bend_dist = _densify_transect_line(
        line_lon, line_lat, spacing_km=spacing_km
    )

    nz = array3d.shape[0]
    ns = s_lon.size
    section = np.full((nz, ns), np.nan, dtype=float)

    fine_lat_1d = np.linspace(-90, 90, newlat)
    fine_lon_1d = np.linspace(-180, 180, newlon)

    irow = np.abs(fine_lat_1d[:, None] - s_lat[None, :]).argmin(axis=0)
    band_masks = (
        np.abs(_wrap_lon_diff_deg(fine_lon_1d[None, :], s_lon[:, None]))
        <= float(lon_half_width_deg)
    )

    # Interpolate only the fine-grid cells the band means will sample.  The
    # flattened indices retain the original latitude/longitude grid positions,
    # including stations that share cells or have no longitude cells in band.
    station_flat = [
        irow[j] * newlon + np.flatnonzero(band_masks[j]) for j in range(ns)
    ]
    query_flat = np.unique(np.concatenate(station_flat))
    if query_flat.size:
        query_points = np.column_stack((
            fine_lat_1d[query_flat // newlon],
            fine_lon_1d[query_flat % newlon],
        ))
        station_indices = [np.searchsorted(query_flat, flat) for flat in station_flat]
        pts = np.column_stack((latmesh.ravel(), lonmesh.ravel()))
        # Use griddata's own nearest-neighbour tie handling once, then look up
        # each depth level's geometry mask by the selected source indices.
        nearest_source = griddata(
            (latmesh.ravel(), lonmesh.ravel()),
            np.arange(latmesh.size, dtype=float),
            query_points,
            method="nearest",
        ).astype(np.intp)

        for k in range(nz):
            z2d = ma.masked_invalid(array3d[k, :, :])
            geom_mask = ma.getmaskarray(z2d).ravel()[nearest_source]

            v = z2d.filled(np.nan).ravel()
            good = np.isfinite(v)

            if np.count_nonzero(good) < 3:
                continue

            interp_lin = griddata(
                pts[good], v[good], query_points, method="linear"
            )
            interp_near = griddata(
                pts[good], v[good], query_points, method="nearest"
            )
            interp_combined = np.where(np.isfinite(interp_lin), interp_lin, interp_near)
            vals = ma.masked_where(geom_mask, interp_combined).filled(np.nan)

            for j, indices in enumerate(station_indices):
                row_vals = vals[indices]
                row_mask = geom_mask[indices]
                valid = (~row_mask) & np.isfinite(row_vals)
                if np.any(valid):
                    section[k, j] = np.nanmean(row_vals[valid])

    return {
        "section": ma.masked_invalid(section),
        "distance_km": s_dist,
        "depth": np.asarray(depth, dtype=float),
        "sample_lon": s_lon,
        "sample_lat": s_lat,
        "bend_distance_km": bend_dist,
        "bend_lon": line_lon,
        "bend_lat": line_lat,
        "lon_half_width_deg": float(lon_half_width_deg),
    }


def cumulative_distance_km(lon_values, lat_values):
    lon_values = np.asarray(lon_values, dtype=float)
    lat_values = np.asarray(lat_values, dtype=float)
    distances = [0.0]
    for index in range(1, len(lon_values)):
        distances.append(
            distances[-1]
            + great_circle_distance(
                lat_values[index - 1],
                lon_values[index - 1],
                lat_values[index],
                lon_values[index],
            )
        )
    return np.asarray(distances, dtype=float)


def _interp_along_segment(value, v0, v1, d0, d1, extrapolate=False):
    if abs(v1 - v0) < 1e-12:
        return d0
    frac = (value - v0) / (v1 - v0)
    if not extrapolate:
        frac = float(np.clip(frac, 0.0, 1.0))
    return d0 + frac * (d1 - d0)


def distance_along_transect_at_site(point_lon, point_lat, line_lon, line_lat):
    """Map a site to distance along the transect using its actual coordinates."""
    line_lon = wrap_longitude_180(line_lon)
    line_lat = np.asarray(line_lat, dtype=float)
    point_lon = float(point_lon)
    point_lat = float(point_lat)
    seg_dist = cumulative_distance_km(line_lon, line_lat)

    if point_lat < line_lat[0]:
        return _interp_along_segment(
            point_lat, line_lat[0], line_lat[1], seg_dist[0], seg_dist[1], extrapolate=True
        )
    if point_lat > line_lat[-1]:
        return _interp_along_segment(
            point_lat, line_lat[-2], line_lat[-1], seg_dist[-2], seg_dist[-1], extrapolate=True
        )

    best_distance = None
    best_lon_gap = np.inf
    for index in range(len(line_lat) - 1):
        lat0, lat1 = line_lat[index], line_lat[index + 1]
        if point_lat < min(lat0, lat1) - 1e-9 or point_lat > max(lat0, lat1) + 1e-9:
            continue
        lon0, lon1 = line_lon[index], line_lon[index + 1]
        if abs(lat1 - lat0) < 1e-9:
            distance = _interp_along_segment(
                point_lon, lon0, lon1, seg_dist[index], seg_dist[index + 1]
            )
            lon_gap = min(lon_abs_diff_deg(point_lon, lon0), lon_abs_diff_deg(point_lon, lon1))
        else:
            distance = _interp_along_segment(
                point_lat, lat0, lat1, seg_dist[index], seg_dist[index + 1]
            )
            lon_gap = lon_abs_diff_deg(point_lon, 0.5 * (lon0 + lon1))
        if lon_gap < best_lon_gap:
            best_lon_gap = lon_gap
            best_distance = distance
    return best_distance if best_distance is not None else seg_dist[0]


def project_points_to_transect(
    points_df,
    line_lon,
    line_lat,
    value_col,
    longitude_window_deg=10.0,
):
    line_lon = wrap_longitude_180(line_lon)

    projected_rows = []
    for _, row in points_df.iterrows():
        point_lon = float(row["Longitude_wrapped"])
        point_lat = float(row["Latitude"])
        if np.min(lon_abs_diff_deg(point_lon, line_lon)) > longitude_window_deg:
            continue

        distance_along_line = distance_along_transect_at_site(
            point_lon, point_lat, line_lon, line_lat
        )

        projected_rows.append(
            {
                "distance_km": distance_along_line,
                "depth_m": float(row["Water depth (m)"]),
                "value": row[value_col],
                "Site name": row["Site name"],
                "Longitude_wrapped": point_lon,
                "Latitude": point_lat,
            }
        )

    return pd.DataFrame(projected_rows)
