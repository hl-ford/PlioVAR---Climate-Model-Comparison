"""Lat-lon, lat-depth and transect plotting functions."""

import numpy as np
import numpy.ma as ma

from .gridding import KNN_implementation
from .gridding import compress_longitude
from .gridding import connect_atlantic
from .gridding import create_mesh
from .gridding import get_bottom
from .gridding import interpolate
from .gridding import interpolate_to_common_grid
from .masks import apply_atlantic_mask_2d
from .masks import apply_atlantic_mask_3d
from .masks import apply_mask


def latlonplot(lat, lon, array, normalise = True, mesh = False, newlat = 1000, newlon = 1000):
    '''
    Input: lat, lon, d18O array. normalise = False if normalising is not needed. Mesh = True if lat and lon input are already meshes
    Output: Benthic array, lat, lon
    '''
    if mesh is False:
        latmesh, lonmesh = create_mesh(lat, lon)
    else:
        latmesh, lonmesh = lat, lon
    connected_lon = connect_atlantic(lonmesh)
    benthic_array = get_bottom(array)
    
    
    interpolate_shape, fine_lat_shape, fine_lon_shape = interpolate(latmesh, connected_lon, benthic_array, newlat = newlat, newlon = newlon)
    masked_shape = apply_atlantic_mask_2d(interpolate_shape, fine_lat_shape, fine_lon_shape)
    
    mask = ma.getmask(masked_shape)
    
    
    imputed_array = KNN_implementation(benthic_array, latmesh, connected_lon)
    
    interpolate_KNN, fine_lat_KNN, fine_lon_KNN = interpolate(latmesh, connected_lon, imputed_array, newlat = newlat, newlon = newlon)
    
    interpolated_masked_KNN = ma.masked_where(mask, interpolate_KNN)
    interpolated_masked_KNN[mask] = np.nan
    
    if normalise:
        mean = np.nanmean(interpolated_masked_KNN)
        std = np.nanstd(interpolated_masked_KNN)
        
        # Normalize the data
        interpolated_masked_KNN = (interpolated_masked_KNN - mean) / std
    
    return interpolated_masked_KNN, fine_lat_KNN, fine_lon_KNN


def shade_below_bathymetry(axis, x, depth, field, color="#d3d3d3"):
    """Shade the seabed inferred from a masked model section.

    The deepest valid cell in each column locates the bottom.  The boundary is
    placed halfway to the next depth level, so missing surface values and other
    gaps remain unshaded.  Where the final model level is still wet, that level
    is the bathymetry boundary and shading starts there.
    """
    values = ma.asarray(field, dtype=float).filled(np.nan)
    x = np.asarray(x, dtype=float)
    depth = np.asarray(depth, dtype=float)
    x_1d = x[0, :] if x.ndim == 2 else x
    depth_1d = depth[:, 0] if depth.ndim == 2 else depth

    if values.shape != (depth_1d.size, x_1d.size):
        raise ValueError("field, x and depth must describe the same section")
    if x_1d.size < 2:
        raise ValueError("section must contain at least two columns")
    if depth_1d.size < 2 or not np.all(np.diff(depth_1d) > 0):
        raise ValueError("depth must increase downwards")

    valid = np.isfinite(values)
    last_wet = np.where(valid, np.arange(depth_1d.size)[:, None], -1).max(axis=0)
    seabed = np.full(x_1d.size, np.nan)
    has_bottom = last_wet >= 0
    above_final_level = has_bottom & (last_wet < depth_1d.size - 1)
    indices = last_wet[above_final_level]
    seabed[above_final_level] = (depth_1d[indices] + depth_1d[indices + 1]) / 2
    seabed[last_wet == depth_1d.size - 1] = depth_1d[-1]

    # Use column edges so even an isolated shallow column is shaded.
    x_edges = np.empty(x_1d.size + 1)
    x_edges[1:-1] = (x_1d[:-1] + x_1d[1:]) / 2
    x_edges[0] = x_1d[0] - (x_1d[1] - x_1d[0]) / 2
    x_edges[-1] = x_1d[-1] + (x_1d[-1] - x_1d[-2]) / 2

    # Two vertices per column keep isolated shallow columns visible without
    # interpolating the seabed across columns that have no known bottom.
    x_steps = np.repeat(x_edges, 2)[1:-1]
    seabed_steps = np.repeat(seabed, 2)

    # Draw behind the model colours, extending past any later y-limit.
    return axis.fill_between(
        x_steps, seabed_steps, depth_1d[-1] + np.ptp(depth_1d),
        where=np.repeat(has_bottom, 2),
        color=color, linewidth=0, zorder=0,
    )


def latdepthplot(lat, lon, depth, array, normalise = True, mesh = False, trimtop = False, top = 0, mask = None, mask_lat = None, mask_lon = None):
    if mesh is False:
        latmesh, lonmesh = create_mesh(lat, lon)
    else:
        latmesh, lonmesh = lat, lon
    connected_lon = connect_atlantic(lonmesh)
    if mask is None:
        masked_array = apply_atlantic_mask_3d(array, latmesh, connected_lon, depth)
    else:
        masked_array = apply_mask(
            array,
            data_lat=latmesh,
            data_lon=connected_lon,
            mask=mask,
            mask_lat=mask_lat,
            mask_lon=mask_lon
        )
    compressed = compress_longitude(masked_array)

    if lat.ndim == 2:
        compressed_lat = compress_longitude(lat)
    else:
        compressed_lat = lat
    
    depth_dl, lat_dl = create_mesh(depth, compressed_lat)
    interpolate_shape, fine_depth_shape, fine_lat_shape = interpolate(depth_dl, lat_dl, compressed, grid = False)
    
    mask = ma.getmask(interpolate_shape)
    
    imputed_array = KNN_implementation(compressed, depth_dl, lat_dl)
    
    interpolate_KNN, fine_depth_KNN, fine_lat_KNN = interpolate(depth_dl, lat_dl, imputed_array, grid = False)
        
    interpolated_masked_KNN = ma.masked_where(mask, interpolate_KNN)
    interpolated_masked_KNN[mask] = np.nan
    
    if trimtop is True:
        interpolated_masked_KNN = np.where(fine_depth_KNN < top, np.nan, interpolated_masked_KNN)
    
    if normalise:
        mean = np.nanmean(interpolated_masked_KNN)
        std = np.nanstd(interpolated_masked_KNN)
        
        # Normalize the data
        interpolated_masked_KNN = (interpolated_masked_KNN - mean) / std
    
    return interpolated_masked_KNN, fine_lat_KNN, fine_depth_KNN


def temperatureadjustedplot(lat, lon, depth, d18Oarray, temparray, normalise = True, mesh = False, trimtop = False, top = 0, samegrid = True, templat = None, templon = None, tempmesh = False, mask = None, mask_lat = None, mask_lon = None):
    '''
    input: lat, lon, depth, d18Oarray, temparray
    output: temp adjusted d18O array, 
    '''       
    if samegrid == True:
        tempmesh = mesh
        templat = lat
        templon = lon
        
    d18OKNN, latKNN, depthKNN = latdepthplot(lat, lon, depth, d18Oarray, normalise = False, mesh = mesh, trimtop = trimtop, top = top, mask = mask, mask_lat = mask_lat, mask_lon = mask_lon)
    tempKNN, templatKNN, tempdepthKNN = latdepthplot(templat, templon, depth, temparray, normalise = False, mesh = tempmesh, trimtop = trimtop, top = top, mask = mask, mask_lat = mask_lat, mask_lon = mask_lon)
    
    same_section_grid = (
        np.array_equal(latKNN, templatKNN)
        and np.array_equal(depthKNN, tempdepthKNN)
    )
    if same_section_grid:
        lat, depth = latKNN, depthKNN
    elif samegrid == False:
        lat, depth, d18OKNN, tempKNN = interpolate_to_common_grid(
            latKNN, depthKNN, d18OKNN, templatKNN, tempdepthKNN, tempKNN
        )
    
    adjusted_array = - 0.224 * tempKNN + 3.26 + d18OKNN
       
    if normalise:
        mean = np.nanmean(adjusted_array)
        std = np.nanstd(adjusted_array)
        
        adjusted_array = (adjusted_array - mean) / std
    
    return adjusted_array, lat, depth


def equatorplot(lat, lon, depth, array, latrange=[-10, 10], normalise=True, mesh=False, trimtop=False, top=0, mask = None, mask_lat = None, mask_lon = None):
    # Wrap lon to [-180, 180)
    lon = (lon + 180) % 360 - 180

    # Build/accept meshes
    if not mesh:
        latmesh, lonmesh = create_mesh(lat, lon)   # (lat, lon) meshes
    else:
        latmesh, lonmesh = lat, lon

    connected_lon = connect_atlantic(lonmesh)
    if mask is None:
        masked_array = apply_atlantic_mask_3d(array, latmesh, connected_lon, depth)
    else:
        masked_array = apply_mask(
            array,
            data_lat=latmesh,
            data_lon=connected_lon,
            mask=mask,
            mask_lat=mask_lat,
            mask_lon=mask_lon
        )
    # Select latitude ROWS in range (works for 2D meshes)
    if latmesh.ndim == 2:
        row_mask = (latmesh[:, 0] >= latrange[0]) & (latmesh[:, 0] <= latrange[1])
    else:
        row_mask = (latmesh >= latrange[0]) & (latmesh <= latrange[1])

    if not np.any(row_mask):
        raise ValueError("No latitude points found within specified range.")

    # Average -10..10 into ONE band -> (depth, lon)
    equator_slice = ma.mean(masked_array[:, row_mask, :], axis=1)

    # Build a 1D lon axis safely
    if lonmesh.ndim == 2:
        lon_1d = connected_lon[0, :]
    else:
        lon_1d = connected_lon

    depth_dl, lon_dl = create_mesh(depth, lon_1d)  # meshes aligned with equator_slice

    interpolate_shape, fine_depth_shape, fine_lon_shape = interpolate(depth_dl, lon_dl, equator_slice, grid=False)
    mask = ma.getmask(interpolate_shape)

    imputed_array = KNN_implementation(equator_slice, depth_dl, lon_dl)
    interpolate_KNN, fine_depth_KNN, fine_lon_KNN = interpolate(depth_dl, lon_dl, imputed_array, grid=False)

    interpolated_masked_KNN = ma.masked_where(mask, interpolate_KNN)
    interpolated_masked_KNN[mask] = np.nan

    if trimtop:
        interpolated_masked_KNN = np.where(fine_depth_KNN < top, np.nan, interpolated_masked_KNN)

    if normalise:
        mean = np.nanmean(interpolated_masked_KNN)
        std = np.nanstd(interpolated_masked_KNN)
        interpolated_masked_KNN = (interpolated_masked_KNN - mean) / std

    return interpolated_masked_KNN, fine_lon_KNN, fine_depth_KNN    


def tempequatorplot(lat, lon, depth, d18Oarray, temparray, normalise=True, mesh=False, trimtop=False, top=0, samegrid=True, templat=None, templon=None, tempmesh=False, latrange = [-10, 10], mask=None, mask_lat=None, mask_lon=None):
    '''
    Adjusts d18O values based on temperature and returns the normalised, optionally trimmed array.
    '''
    # Step 1: Check if same grid should be used
    if samegrid:
        tempmesh = mesh
        templat = lat
        templon = lon

    # Step 2: Generate initial plots for d18O and temperature arrays
    d18OKNN, latKNN, lonKNN = equatorplot(lat, lon, depth, d18Oarray, normalise=False, mesh=mesh, latrange = latrange, mask = mask, mask_lat = mask_lat, mask_lon = mask_lon)
    tempKNN, templatKNN, templonKNN = equatorplot(templat, templon, depth, temparray, normalise=False, mesh=tempmesh, latrange = latrange, mask = mask, mask_lat = mask_lat, mask_lon = mask_lon)

    # Step 3: Interpolate to a common grid if required
    if not samegrid:
        lat, lon, d18OKNN, tempKNN = interpolate_to_common_grid(latKNN, lonKNN, d18OKNN, templatKNN, templonKNN, tempKNN)
    else:
        # Ensure both grids are aligned
        if np.array_equal(latKNN, templatKNN) and np.array_equal(lonKNN, templonKNN):
            lat, lon = latKNN, lonKNN
        else:
            raise ValueError("Grids do not match despite 'samegrid=True'.")

    # Step 4: Adjust d18O values based on temperature
    adjusted_array = -0.224 * tempKNN + 3.26 + d18OKNN

    # Step 5: Optional - Trim top of the array based on specified depth
    if trimtop:
        # Generate an interpolated depth grid that matches adjusted_array's shape
        depth_grid = np.linspace(depth.min(), depth.max(), adjusted_array.shape[0])

        # Create a mask for rows where depth_grid < top
        depth_mask = depth_grid < top

        # Apply the mask to set rows to NaN
        adjusted_array[depth_mask, :] = np.nan
    # Step 6: Optional - Normalise the adjusted array
    if normalise:
        mean = np.nanmean(adjusted_array)
        std = np.nanstd(adjusted_array)
        adjusted_array = (adjusted_array - mean) / std

    return adjusted_array, lat, lon


def _join_labels(names):
    """Combine labels of overlapping points: consecutive integers become a range ("925–929")."""
    names = [str(n) for n in names]
    if len(names) > 2 and all(n.isdigit() for n in names):
        nums = sorted(int(n) for n in names)
        if nums == list(range(nums[0], nums[-1] + 1)):
            return f"{nums[0]}–{nums[-1]}"
    return "/".join(names)


def place_point_labels(ax, x, y, texts, transform=None, fontsize=10, marker_size=36,
                       avoid_lines=(), radii=(4, 12, 24, 40, 60, 85, 115, 150, 190, 240),
                       pad=2, merge_overlapping=True, text_kw=None, leader_kw=None):
    """Label scatter points without overlaps, as close to each point as possible.

    Each label tries positions around its point, nearest first (right, left, diagonals,
    above, below; then wider rings), and takes the first one that stays inside the axes
    and does not overlap another label, any marker or any of ``avoid_lines``, and whose
    leader line would not cross another leader or run through another label. If nothing
    is completely free, the position with the smallest penalty is used. Labels beyond the
    first ring get a thin leader line to their point. The result is deterministic.

    x, y               point coordinates in ``transform`` (default: ax.transData)
    marker_size        scatter ``s`` (points**2), used to keep labels off the markers
    avoid_lines        sequence of (xs, ys) polylines in ``transform`` coordinates
    radii              gaps (points) between marker edge and label, tried in order
    pad                clearance (points) kept around each label, e.g. to cover a text bbox
    merge_overlapping  share one label ("607/U1313") between points closer than one marker
                       radius: True for all such points, False for none, or a list of label
                       sets, e.g. [{"607", "U1313"}], to merge only those
    Returns the list of Annotation objects.
    """
    fig = ax.figure
    transform = ax.transData if transform is None else transform
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    marker_r_pt = np.sqrt(marker_size / np.pi)
    r_px = marker_r_pt * px_per_pt
    pad_px = pad * px_per_pt
    clump_px = 2.5 * r_px  # dots this close overlap on the map

    x, y = np.asarray(x, float), np.asarray(y, float)
    all_pts = transform.transform(np.column_stack([x, y]))
    markers = [(px - r_px, py - r_px, px + r_px, py + r_px) for px, py in all_pts]

    # group points that sit on top of each other
    def may_merge(a, b):
        if merge_overlapping is True or merge_overlapping is False:
            return merge_overlapping
        return any({str(texts[a]), str(texts[b])} <= {str(t) for t in group} for group in merge_overlapping)

    groups = []
    for i, p in enumerate(all_pts):
        for g in groups:
            if may_merge(g[0], i) and np.hypot(*(all_pts[g[0]] - p)) < r_px:
                g.append(i)
                break
        else:
            groups.append([i])
    labels = [_join_labels([texts[i] for i in g]) for g in groups]
    gx = np.array([x[g].mean() for g in groups])
    gy = np.array([y[g].mean() for g in groups])
    pts = transform.transform(np.column_stack([gx, gy]))

    line_pts = []
    for xs, ys in avoid_lines:
        xy = transform.transform(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]))
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 3))
            line_pts.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
    line_pts = np.vstack(line_pts) if line_pts else np.empty((0, 2))

    axes_box = ax.get_window_extent(renderer)
    # (dx, dy, ha, va) unit directions, horizontal first
    directions = [(1, 0, "left", "center"), (-1, 0, "right", "center"),
                  (0.75, 0.75, "left", "bottom"), (-0.75, 0.75, "right", "bottom"),
                  (0.75, -0.75, "left", "top"), (-0.75, -0.75, "right", "top"),
                  (0, 1, "center", "bottom"), (0, -1, "center", "top"),
                  (0.9, 0.45, "left", "bottom"), (-0.9, 0.45, "right", "bottom"),
                  (0.9, -0.45, "left", "top"), (-0.9, -0.45, "right", "top")]
    # shallow angles: lets labels near a crowded edge stack sideways with near-parallel leaders
    directions += [(sx, sy, "left" if sx > 0 else "right", "center")
                   for sy in (0.15, -0.15, 0.3, -0.3) for sx in (1, -1)]

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def crosses(p1, p2, q1, q2):
        def orient(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))

    def segment_hits_box(p1, p2, b, n=24):
        t = np.linspace(0.05, 0.95, n)[:, None]
        s = p1 + t * (p2 - p1)
        return bool(np.any((s[:, 0] > b[0]) & (s[:, 0] < b[2]) & (s[:, 1] > b[1]) & (s[:, 1] < b[3])))

    # Candidate positions per label, with the costs that don't depend on placement order.
    # Penalties, worst first: label on label > label on a marker = label on/under a leader
    #   = leader through a label > crossed leaders > leader past a marker > label on a line.
    probe = ax.annotate("", xy=(0, 0), xycoords=transform, xytext=(0, 0),
                        textcoords="offset points", fontsize=fontsize, **(text_kw or {}))
    candidates = []
    for i in range(len(groups)):
        own = set(groups[i])
        probe.set_text(labels[i])
        probe.xy = (gx[i], gy[i])
        cands = []
        for ring, gap in enumerate(radii):
            off = marker_r_pt + gap
            for dx, dy, ha, va in directions:
                probe.set_position((dx * off, dy * off))
                probe.set_ha(ha)
                probe.set_va(va)
                bb = probe.get_window_extent(renderer)
                box = (bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)
                if (box[0] < axes_box.x0 or box[2] > axes_box.x1
                        or box[1] < axes_box.y0 or box[3] > axes_box.y1):
                    continue
                cost = sum(1e6 + overlap(box, m) for m in markers if overlap(box, m) > 0)
                if len(line_pts):
                    inside = ((line_pts[:, 0] > box[0]) & (line_pts[:, 0] < box[2])
                              & (line_pts[:, 1] > box[1]) & (line_pts[:, 1] < box[3]))
                    cost += 50.0 * inside.sum()
                leader = None
                if ring > 0:
                    # matplotlib draws the leader towards the label centre (clipped at the label box)
                    anchor = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    leader = (pts[i], anchor)
                    cost += 1e4 * sum(segment_hits_box(pts[i], anchor, m)
                                      for k, m in enumerate(markers)
                                      if k not in own and np.hypot(*(all_pts[k] - pts[i])) > clump_px)
                cost += 100.0 * ring  # nearer is better: each extra ring costs a little
                cands.append((cost, (dx * off, dy * off), ha, va, box, ring, leader))
        if not cands:  # nothing fits inside the axes: put it right of the point
            cands.append((0.0, (marker_r_pt + radii[0], 0), "left", "center", None, 0, None))
        candidates.append(cands)
    probe.remove()

    def pair_cost(c, d):
        """Penalty between two chosen label positions (symmetric)."""
        _, _, _, _, box_c, _, lead_c = c
        _, _, _, _, box_d, _, lead_d = d
        cost = 0.0
        if box_c is not None and box_d is not None and overlap(box_c, box_d) > 0:
            cost += 1e7 + overlap(box_c, box_d)
        if lead_d is not None and box_c is not None and segment_hits_box(lead_d[0], lead_d[1], box_c):
            cost += 1e6
        if lead_c is not None and box_d is not None and segment_hits_box(lead_c[0], lead_c[1], box_d):
            cost += 1e6
        if (lead_c is not None and lead_d is not None and np.hypot(*(lead_c[0] - lead_d[0])) > clump_px
                and crosses(lead_c[0], lead_c[1], lead_d[0], lead_d[1])):
            cost += 1e5
        return cost

    def layout(order):
        """Greedy placement in ``order``, then repeated single-label moves while the total improves."""
        chosen = {}
        for i in order:
            best = None
            for cand in candidates[i]:
                cost = cand[0] + sum(pair_cost(cand, chosen[j]) for j in chosen)
                if best is None or cost < best[0]:
                    best = (cost, cand)
                if cost < 1e3:
                    break
            chosen[i] = best[1]
        for _ in range(6):
            moved = False
            for i in order:
                others = [chosen[j] for j in chosen if j != i]
                current = chosen[i][0] + sum(pair_cost(chosen[i], o) for o in others)
                for cand in candidates[i]:
                    cost = cand[0] + sum(pair_cost(cand, o) for o in others)
                    if cost < current - 1e-9:
                        chosen[i], current, moved = cand, cost, True
            if not moved:
                break
        keys = list(chosen)
        total = sum(chosen[k][0] for k in keys) + sum(
            pair_cost(chosen[a], chosen[b]) for n, a in enumerate(keys) for b in keys[n + 1:])
        return total, chosen

    # Greedy placement depends on order, so try a few fixed orders and keep the best layout.
    dist = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    crowding = (dist < 60 * px_per_pt).sum(axis=1)
    orders = [np.argsort(-crowding, kind="stable"), np.argsort(crowding, kind="stable"),
              np.argsort(-pts[:, 1], kind="stable"), np.argsort(pts[:, 1], kind="stable"),
              np.argsort(-pts[:, 0], kind="stable"), np.argsort(pts[:, 0], kind="stable")]
    _, chosen = min((layout(o) for o in orders), key=lambda result: result[0])

    text_kw = {"color": "0.15", **(text_kw or {})}
    leader_kw = {"arrowstyle": "-", "color": "0.35", "linewidth": 0.8, **(leader_kw or {})}
    annotations = []
    for i in range(len(groups)):
        _, pos, ha, va, _, ring, _ = chosen[i]  # (static cost, offset, ha, va, box, ring, leader)
        arrow = {"arrowprops": {"shrinkA": 1, "shrinkB": marker_r_pt, **leader_kw}} if ring > 0 else {}
        annotations.append(ax.annotate(labels[i], xy=(gx[i], gy[i]), xycoords=transform, xytext=pos,
                                       textcoords="offset points", fontsize=fontsize, ha=ha, va=va,
                                       zorder=7, **arrow, **text_kw))
    return annotations
