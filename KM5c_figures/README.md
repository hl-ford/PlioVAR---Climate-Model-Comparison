# KM5c benthic δ¹⁸O and δ¹³C: model–proxy comparison figures

Code to reproduce the figures and summary tables in *[paper title, authors, journal, DOI]*.

The notebooks compare isotope-enabled climate model simulations of the KM5c interglacial (~3.205 Ma) and the pre-industrial with benthic foraminiferal δ¹⁸O and δ¹³C from PlioVAR (KM5c) and OC3 (modern core tops).

| Model | Variables used |
|---|---|
| iCESM1.2-ITPCAS | δ¹⁸O, temperature, streamfunction |
| iCESM1.3 | δ¹⁸O, temperature, streamfunction |
| HadCM3 | δ¹⁸O, temperature, streamfunction |
| CESM1.2.2.1 | δ¹³C (DIC), temperature, streamfunction |

## Repository layout

```
km5c/                         shared code, imported by the notebooks
  paths.py                    all input/output locations (edit here, or set KM5C_DATA_DIR)
  models.py                   load_icesm12(), load_icesm13(), load_hadcm3(), load_cesm122(): cached model loaders
  proxies.py                  core-top and PlioVAR processing, site matching, comparison tables
  gridding.py, masks.py       regridding/interpolation and ocean-basin masks
  plotting.py, transects.py   lat–depth/transect plots and site projection onto transects
  stats.py                    Atlantic (>200 m) area/volume-weighted statistics
  weighted_knn_interpolation.py
notebooks/                    one notebook per figure group (run from this folder)
outputs/figures/              PNGs written by the notebooks
outputs/tables/               CSV tables written by km5c.proxies and Summary_statistics.ipynb
```

Importing `km5c` reads no data. Each notebook's first cell imports only what it uses and calls only the loaders it needs. For example, `d13C.ipynb` loads CESM1.2.2.1 only, and `Site_locations.ipynb` loads only the proxy tables.

### Core-top comparison (`km5c.proxies`)
For each PlioVAR site, the core tops (OC3; Schmittner et al., 2017) are searched within 500 km great-circle distance and ±10% of the site's water depth. If none are found, a 1000 km / ±20% search is tried. In the KM5c window (3.185–3.225 Ma), the site's own core-top samples (< 6 ka) are added as one extra site. The core-top s.d. is the pooled s.d. of the matched sites, each weighted equally: the mean within-site variance plus the variance of the site means. Taxon offsets, age windows and search radii are constants at the top of `proxies.py`.

## Notebooks

Figures are saved to `outputs/figures/` as PNG at 300 dpi.

| Notebook | Content | Saved figure(s) |
|---|---|---|
| `Site_locations.ipynb` | Proxy site locations: lat–lon map and lat–depth section | `site_locations.png` |
| `maps.ipynb` | Transect location maps | `maps_{d13C,d18O}_{A13,A16,equator}.png` |
| `Proxies.ipynb` | mPWP proxy-only lat–depth δ¹³C and δ¹⁸O; PRISM3D vs KM5c proxies (appendix) | `proxies_KM5c_lat_depth.png`, `proxies_PRISM3D_vs_KM5c.png` |
| `Modern_OC3.ipynb` | Modern-ocean lat–depth δ¹⁸O and δ¹³C compared with OC3 | `modern_OC3_d18O_d13C.png` |
| `d13C.ipynb` | Model δ¹³C lat–depth sections | `d13C_lat_depth.png` |
| `d13C_with_points.ipynb` | Model δ¹³C lat–depth sections with proxy points | `d13C_with_points.png` |
| `d13C_transects.ipynb` | Model δ¹³C along the A16, A13 and Equator transects | `d13C_transects.png` |
| `d18O.ipynb` | Model δ¹⁸O lat–depth sections: pre-industrial vs OC3, KM5c vs PlioVAR | `d18O_lat_depth.png` |
| `d18O_calcite.ipynb` | 3×3 δ¹⁸O(seawater), temperature, δ¹⁸O(calcite) panels (appendix) | `d18O_calcite_KM5c_3x3.png`, `d18O_calcite_PI_3x3.png` |
| `d18O_transects.ipynb` | Model δ¹⁸O along the A13, A16 and Equator transects | `d18O_transects.png` |
| `SST.ipynb` | SST anomaly for each model vs PlioVAR KM5c SST (appendix) | `SST_anomaly.png` |
| `Streamfunctions.ipynb` | Atlantic meridional overturning streamfunction, pre-industrial and Pliocene, one model per row | `moc_PI_Pliocene_4x2.png` |
| `Summary_statistics.ipynb` | Atlantic (>200 m) summary statistics and model–proxy tables | (tables: `outputs/tables/atlantic_below200m_*.csv`) |
| `Transect_site_locations.ipynb` | Placeholder; merged into `Site_locations.ipynb` | — |

## Input data

The input data are **not included** here. Model output was obtained from the modelling groups; the observational and proxy datasets are available from their original publishers. Put them under `data/raw/`, or set `KM5C_DATA_DIR` to another folder, using the layout documented at the top of [`km5c/paths.py`](km5c/paths.py).

| Folder | Dataset | Source |
|---|---|---|
| `iCESM1.2-ITPCAS/` | KM5c (Eoi400) and 0 ka isotope-enabled simulations | *[reference]* |
| `iCESM1.3/` | PlioMIP2 and pre-industrial isotope-enabled simulations | *[reference]* |
| `HadCM3/` | 3.205 Ma and pre-industrial simulations; `basin_hadcm3` basin index file | *[reference]* |
| `CESM1.2.2.1/` | Pliocene and pre-industrial carbon-isotope simulations | *[reference]* |
| `PlioVAR_benthic/` | PlioVAR benthic δ¹⁸O/δ¹³C compilation (QC-checked) and site depths | *[reference]* |
| `PlioVAR_SST/` | PlioVAR KM5c SST compilation | *[reference / PANGAEA DOI]* |
| `Schmittner2017/` | OC3 core-top benthic δ¹³C and δ¹⁸O compilation | Schmittner et al. (2017) *[full reference]* |
| `GEBCO/` | GEBCO 2024 sub-ice topography grid | GEBCO Compilation Group (2024) |
| `WOA23/` | World Ocean Atlas 2023 annual temperature, 1° | Locarnini et al. (2023) *[full reference]* |
| `modern_d13C/` | Gridded modern δ¹³C(DIC) climatology | *[reference]* |
| `modern_d18O/` | Gridded seawater δ¹⁸O | *[reference]* |

## Running

```bash
pip install -r requirements.txt          # Python 3.11; cartopy is easiest to get via conda
export KM5C_DATA_DIR=/path/to/data       # optional; defaults to ./data/raw
cd notebooks
jupyter notebook                          # or: jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

Loaders are cached within a session, so a model is read and regridded at most once per kernel. `comparison_tables()` writes the proxy CSVs to `outputs/tables/` the first time it is called.

## Licence

Code: MIT (see `LICENSE`).
