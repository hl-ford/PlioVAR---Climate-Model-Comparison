"""Benthic proxy processing and the PlioVAR vs core-top (OC3) comparison tables.

Pipeline:
    1. ``core_top_means``     Schmittner et al. (2017) OC3 core tops, averaged per core
    2. ``pliovar_site_means`` PlioVAR benthic records, averaged per site within an age window
    3. ``match_core_tops``    nearby core tops for each PlioVAR site (conservative, then liberal search)
    4. ``comparison_table``   PlioVAR minus core-top anomaly table

``comparison_tables()`` runs all of it once (cached) and writes the CSVs to ``outputs/tables``.
"""

import math
import os
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .masks import atlantic_polygon, southern_polygon
from .paths import PLIOVAR_CHECKED_DIR, PLIOVAR_SITE_DEPTHS_PATH, SCHMITTNER_DIR, TABLE_DIR
from .transects import great_circle_distance

# Benthic taxon flags. Cibicidoides/Planulina are used as measured; for d18O,
# Uvigerina and Hydroides elegans are offset to Cibicidoides-equivalent values.
CIBICIDOIDES_FLAGS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 30]
UVIGERINA_FLAGS = [5, 26]
UVIGERINA_FLAGS_CORE_TOP = [5]  # the core-top file only uses flag 5 for Uvigerina
HYDROIDES_FLAGS = [13]
UVIGERINA_D18O_OFFSET = 0.47
HYDROIDES_D18O_OFFSET = 1.07

# PlioVAR age windows (Ma)
AGE_WINDOWS = {"PLIOMIP1": (3.025, 3.264), "PLIOMIP2": (3.185, 3.225)}
# PlioVAR samples younger than this (Ma) are the site's own core top ("self")
SELF_MAX_AGE_MA = 0.006

# Core-top searches: maximum great-circle distance (km) and depth tolerance as
# site depth / factor. The liberal search is only used when the conservative one finds nothing.
SEARCHES = {"conservative": (500, 10), "liberal": (1000, 5)}

PLIOVAR_COLUMNS = ['Site', 'Lon', 'Lat', 'Age_Ma_orig', 'Age_Ma_PLIOVAR', 'd18O_bf', 'd13C_bf',
                   'd18O_taxon_flag', 'd13C_taxon_flag', 'Depth_mbsf']


def _rows_to_frame(rows, columns=None):
    """Stack row dicts into a DataFrame (same construction and dtypes as the original row-by-row concat)."""
    frames = ([pd.DataFrame(columns=columns)] if columns is not None else []) + [pd.DataFrame([r]) for r in rows]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _benthic_subset(group, flag_col, value_col=None, uvigerina_flags=UVIGERINA_FLAGS):
    """Keep usable benthic taxa.

    For d13C (``value_col`` None) only Cibicidoides/Planulina are kept. For d18O,
    Uvigerina and Hydroides are kept too and offset to Cibicidoides-equivalent values.
    """
    flags = group[flag_col]
    keep = flags.isin(CIBICIDOIDES_FLAGS)
    out = group.copy()
    if value_col is not None:
        uvigerina = flags.isin(uvigerina_flags)
        hydroides = flags.isin(HYDROIDES_FLAGS)
        out.loc[uvigerina, value_col] = out.loc[uvigerina, value_col] - UVIGERINA_D18O_OFFSET
        out.loc[hydroides, value_col] = out.loc[hydroides, value_col] - HYDROIDES_D18O_OFFSET
        keep = keep | uvigerina | hydroides
    return out[keep]


def _taxa_present(group, flag_col):
    taxa = set()
    for flag in group[flag_col]:
        if flag in CIBICIDOIDES_FLAGS:
            taxa.add("Cibicidoides")
        elif flag in UVIGERINA_FLAGS:
            taxa.add("Uvigerina")
        elif flag in HYDROIDES_FLAGS:
            taxa.add("Hydroides")
    return taxa


def core_top_means(core_tops, isotope):
    """Per-core mean and variance of Atlantic core tops for ``isotope`` ('d18O' or 'd13C')."""
    value_col = "D18O" if isotope == "d18O" else "d13C"
    rows = []
    for core, group in core_tops.groupby("Core"):
        group = _benthic_subset(group, "speciesflag", value_col if isotope == "d18O" else None,
                                uvigerina_flags=UVIGERINA_FLAGS_CORE_TOP)
        if group.empty:
            continue
        lon, lat = group["longitude"].mean(), group["latitude"].mean()
        value = group[value_col].mean()
        if math.isnan(value) or not atlantic_polygon.contains_point((lon, lat)):
            continue
        rows.append({"Site": core, "count": len(group),
                     "lon": lon, "lon_var": group["longitude"].var(),
                     "lat": lat, "lat_var": group["latitude"].var(),
                     f"published {isotope}": value, f"published {isotope}_var": group[value_col].var(),
                     "depth": group["sitedepth_m"].mean(), "depth_var": group["sitedepth_m"].var()})
    return _rows_to_frame(rows)


def load_pliovar():
    """All QC-checked PlioVAR benthic records, with Age_Ma_PLIOVAR filled from Age_Ma_orig where missing."""
    age_data = pd.DataFrame()
    for filename in os.listdir(PLIOVAR_CHECKED_DIR):
        if not filename.endswith('.xlsx') or filename.startswith('~$'):
            continue
        df = pd.read_excel(os.path.join(PLIOVAR_CHECKED_DIR, filename))
        try:
            extracted = df[PLIOVAR_COLUMNS].copy()
        except KeyError:
            missing = [key for key in PLIOVAR_COLUMNS[:-1] if key not in df.columns]
            print(f"Error processing file: {filename}. KeyError occurred. Missing keys: {missing}")
            continue
        extracted['Source'] = filename
        extracted = extracted.dropna(subset=['Site'])
        extracted['Site'] = extracted['Site'].apply(lambda x: int(x) if isinstance(x, (int, float)) else x)
        age_data = pd.concat([age_data, extracted], ignore_index=True)
    for index, row in age_data.iterrows():
        if np.isnan(row["Age_Ma_PLIOVAR"]):
            age_data.loc[index, "Age_Ma_PLIOVAR"] = row["Age_Ma_orig"]
    return age_data


def load_site_depths():
    """Water depth (m) of each PlioVAR site, keyed by site name."""
    site_depths = pd.read_csv(PLIOVAR_SITE_DEPTHS_PATH)
    first = site_depths.drop_duplicates("Site", keep="first")
    return dict(zip(first["Site"], first["Site depth"]))


def pliovar_site_means(age_data, site_depths, isotope, window, with_taxa=False):
    """Per-site mean and variance of PlioVAR ``isotope`` within the age ``window`` (Ma), Atlantic + Southern Ocean."""
    value_col, flag_col = f"{isotope}_bf", f"{isotope}_taxon_flag"
    young, old = window
    rows = []
    for _, group in age_data.groupby("Site"):
        group = _benthic_subset(group, flag_col, value_col if isotope == "d18O" else None)
        taxa = _taxa_present(group, flag_col) if with_taxa else None
        age = group["Age_Ma_PLIOVAR"]
        group = group[~((age > old) | (age < young))]  # samples with no age are kept
        if group.empty:
            continue
        lon, lat = group["Lon"].mean(), group["Lat"].mean()
        value = group[value_col].mean()
        if math.isnan(value):
            continue
        if not (atlantic_polygon.contains_point((lon, lat)) or southern_polygon.contains_point((lon, lat))):
            continue
        site = group["Site"].iloc[0]
        row = {"Site": site, "lon": lon, "lon_var": group["Lon"].var(), "lat": lat, "lat_var": group["Lat"].var(),
               f"published {isotope}": value, f"published {isotope}_var": group[value_col].var(),
               "depth": site_depths.get(str(site)), "count": len(group)}
        if with_taxa:
            row["taxons"] = "; ".join(sorted(taxa))  # e.g. "Cibicidoides; Uvigerina"; sorted so CSVs are reproducible
        rows.append(row)
    return _rows_to_frame(rows)


def match_core_tops(lat, lon, depth, core_tops, isotope, search):
    """Core tops within the ``search`` distance and depth tolerance of a site, in table order."""
    maxdist, depth_factor = SEARCHES[search]
    tolerance = depth / depth_factor
    hits = [i for i, ct in core_tops.iterrows()
            if (great_circle_distance(lat, lon, ct["lat"], ct["lon"]) < maxdist)
            & (np.abs(depth - ct["depth"]) < tolerance)]
    return core_tops.loc[hits]


def _search(site, core_tops, isotope):
    """Conservative search, falling back to liberal; returns (matches, search name or None)."""
    for search in SEARCHES:
        matches = match_core_tops(site["lat"], site["lon"], site["depth"], core_tops, isotope, search)
        if len(matches):
            return matches, search
    return core_tops.iloc[0:0], None


def pooled_variance(means, variances, overall_mean):
    """Variance of a group of sites, each weighted equally (law of total variance).

    Mean within-site variance plus the variance of the site means about ``overall_mean``.
    A site with a single sample has no within-site variance (NaN), which counts as 0.
    """
    means = np.asarray(means, dtype=float)
    within = np.nan_to_num(np.asarray(variances, dtype=float), nan=0.0)
    return within.mean() + np.mean((means - overall_mean) ** 2)


def compare_sites(sites, core_tops, isotope, age_data=None):
    """Mean and pooled variance of the matched core tops for every PlioVAR site.

    With ``age_data``, a site's own core-top samples ("self", younger than SELF_MAX_AGE_MA)
    are included and weighted as one extra site. Variance combines each core top's own
    variance with the spread between core tops (see ``pooled_variance``).
    """
    value = f"published {isotope}"
    columns = ["site", "lon", "lat", "depth", f"{isotope}benave", f"{isotope}benvar", f"{isotope}sed", "count",
               "algorithm type"]
    rows, used = [], set()
    for _, site in sites.iterrows():
        matches, search = _search(site, core_tops, isotope)
        used.update(matches["Site"])
        count = len(matches)
        mean = var = None
        algorithm = search if search else "N/A"
        site_means = matches[value].tolist()
        site_vars = matches[f"{value}_var"].tolist()
        if count:
            mean = np.mean(site_means)
        if age_data is not None:
            own = age_data[(age_data["Site"] == site["Site"]) & (age_data["Age_Ma_PLIOVAR"] < SELF_MAX_AGE_MA)]
            if not own.empty:
                self_mean, self_count = own[f"{isotope}_bf"].mean(), 1
                site_means.append(self_mean)
                site_vars.append(own[f"{isotope}_bf"].var())
                if count:
                    mean = (mean * count + self_mean * self_count) / (count + self_count)
                    algorithm = f"{search} + self"
                else:
                    mean, algorithm = self_mean, "self"
        if site_means:
            var = pooled_variance(site_means, site_vars, mean)
        row = {"site": site["Site"], "lon": site["lon"], "lat": site["lat"], "depth": site["depth"],
               f"{isotope}benave": mean, f"{isotope}benvar": var, f"{isotope}sed": site[value],
               "count": count, "algorithm type": algorithm}
        if "taxons" in sites.columns:
            row["taxons"] = site["taxons"]
        rows.append(row)
    return _rows_to_frame(rows, columns), used


def comparison_table(compared, sites, isotope, label, add_average):
    """PlioVAR value, core-top value, both s.d.s and the anomaly for each site."""
    columns = ["Site name", "Longitude", "Latitude", "Water depth (m)", "count", f"{label} {isotope}", f"{label} s.d.",
               f"OC3 {isotope}", "OC3 s.d.", f"{label} search", f"{label} anomaly"]
    rows = []
    for index, row in compared.iterrows():
        plio = row[f"{isotope}sed"]
        core_top, core_top_var = row[f"{isotope}benave"], row[f"{isotope}benvar"]
        entry = {"Site name": row["site"], "Longitude": row["lon"], "Latitude": row["lat"],
                 "Water depth (m)": row["depth"], "count": row["count"],
                 f"{label} {isotope}": plio,
                 f"{label} s.d.": math.sqrt(sites.loc[index, f"published {isotope}_var"]),
                 f"OC3 {isotope}": core_top,
                 "OC3 s.d.": math.sqrt(core_top_var) if core_top_var is not None else None,
                 f"{label} search": row["algorithm type"],
                 f"{label} anomaly": plio - core_top if core_top is not None else None}
        if "taxons" in compared.columns:
            entry["taxons"] = row["taxons"]
        rows.append(entry)
    table = _rows_to_frame(rows, columns)
    if add_average:
        if isotope == "d18O":
            average = table.mean(numeric_only=True).to_dict()
            average.update({"Site name": "Average", f"{label} search": "N/A"})
        else:
            numeric = [c for c in columns if c not in ("Site name", f"{label} search")]
            average = table[numeric].mean()
            average["Site name"] = "Average"
            average[f"{label} search"] = "-"
        table = pd.concat([table, pd.DataFrame([average])], ignore_index=True)
    return table


@cache
def comparison_tables(write=True):
    """Build every proxy table once; with ``write``, save them to outputs/tables.

    Returns a namespace with the core tops (``d18O_oc3``, ``comparison_3d``), the PlioVAR
    records (``age_data``), the per-site means (``PLIOMIP1_d18O`` ...), the matching results
    (``PLIOMIP1_d18O_output`` ...) and the comparison tables (``PLIOMIP2_d18O_comparison_table`` ...).
    PLIOMIP1/PLIOMIP2 are the two PlioVAR age windows in AGE_WINDOWS.
    """
    core_tops = pd.read_csv(f"{SCHMITTNER_DIR}/schmittner2017-coretop(1).csv")
    age_data = load_pliovar()
    site_depths = load_site_depths()
    out = {"age_data": age_data,
           "comparison_3d": core_top_means(core_tops, "d13C"),
           "d18O_oc3": core_top_means(core_tops, "d18O")}
    core_top_of = {"d18O": out["d18O_oc3"], "d13C": out["comparison_3d"]}

    for isotope in ("d18O", "d13C"):
        for label, window in AGE_WINDOWS.items():
            km5c = label == "PLIOMIP2"
            sites = pliovar_site_means(age_data, site_depths, isotope, window, with_taxa=km5c and isotope == "d18O")
            compared, used = compare_sites(sites, core_top_of[isotope], isotope, age_data if km5c else None)
            out[f"{label}_{isotope}"] = sites
            out[f"{label}_{isotope}_output"] = compared
            out[f"{label}_{isotope}_comparison_table"] = comparison_table(
                compared, sites, isotope, label, add_average=not km5c)
            if km5c and isotope == "d18O":
                out["used_sites_df"] = pd.DataFrame(sorted(used), columns=["Site"])

    if write:
        for isotope in ("d18O", "d13C"):
            for n, label in ((1, "PLIOMIP1"), (2, "PLIOMIP2")):
                out[f"{label}_{isotope}_output"].to_csv(TABLE_DIR / f"{isotope}_PlioVAR{n}_comparison.csv", index=False)
                out[f"{label}_{isotope}_comparison_table"].to_csv(
                    TABLE_DIR / f"{isotope}_PlioMIP{n}_comparison_table.csv", index=False)
        out["used_sites_df"].to_csv(TABLE_DIR / "used_OC3_sites.csv", index=False)
    return SimpleNamespace(**out)
