"""Central file locations for the KM5c figure notebooks.

Every input path used by the ``km5c`` package and the notebooks is defined
here. Input data are not distributed with this repository; place them under
the ``Data`` folder next to this project (layout below) or point the ``KM5C_DATA_DIR`` environment
variable at a folder with the same layout.

    <KM5C_DATA_DIR>/
        iCESM1.2-ITPCAS/isotopes/          Delta18O_*, TEMP_* (1x1 monmean)
        iCESM1.2-ITPCAS/streamfunction/    VVEL_*, REGION_MASK_*
        iCESM1.3/isotopes/                 *_climo_1x1deg.nc, *.ann_clim_compress_new.nc
        iCESM1.3/streamfunction/           *_moc_climo_new.nc, *_iPI_dye.003_*_climo.nc
        HadCM3/isotopes/                   *_d18o_and_temp3900_4000.nc
        HadCM3/streamfunction/             *_V_and_W_*.nc, basin_hadcm3
        CESM1.2.2.1/                       *_ciso_T31_gx3v7_ALL.2901-3000_Reg1DegGrid.nc
        PlioVAR_benthic/                   CHECKED/*.xlsx, Site Depths.csv
        PlioVAR_SST/                       PlioVAR-KM5c_SST.tab
        Schmittner2017/                    schmittner2017-coretop(1).csv (OC3 core tops)
        GEBCO/                             GEBCO_2024_sub_ice_topo.nc
        WOA23/                             woa23_decav_t00mn01.csv
        modern_d13C/                       C13_Climatology.nc
        modern_d18O/                       calculated_d18O_v1_1.nc
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# inputs live in the "Data" folder next to "Jupyter notebooks" (override with KM5C_DATA_DIR)
DATA_DIR = Path(os.environ.get("KM5C_DATA_DIR", REPO_ROOT.parent / "Data"))

FIG_DIR = REPO_ROOT / "outputs" / "figures"
TABLE_DIR = REPO_ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Model output: isotopes and temperature
ICESM12_ISOTOPE_DIR = DATA_DIR / "iCESM1.2-ITPCAS" / "isotopes"
ICESM13_ISOTOPE_DIR = DATA_DIR / "iCESM1.3" / "isotopes"
HADCM3_ISOTOPE_DIR = DATA_DIR / "HadCM3" / "isotopes"
CESM122_DIR = DATA_DIR / "CESM1.2.2.1"

# Model output: streamfunctions
ICESM12_SF_DIR = DATA_DIR / "iCESM1.2-ITPCAS" / "streamfunction"
ICESM13_SF_DIR = DATA_DIR / "iCESM1.3" / "streamfunction"
HADCM3_SF_DIR = DATA_DIR / "HadCM3" / "streamfunction"

ICESM12_PLIO_VVEL_PATH = ICESM12_SF_DIR / "VVEL_ISO_Eoi400_last100years_monthly.nc"
ICESM12_PI_VVEL_PATH = ICESM12_SF_DIR / "VVEL_iso_0ka_last100years_monthly.nc"
ICESM12_PLIO_MASK_PATH = ICESM12_SF_DIR / "REGION_MASK_ISO_Eoi400_last100years.nc"
ICESM13_PLIO_MOC_PATH = ICESM13_SF_DIR / "b.ie12.B1850.f19_g16.PMIP4-pliomip2.icesm1.3_ANN_050101_060012_moc_climo_new.nc"
ICESM13_PI_MOC_PATH = ICESM13_SF_DIR / "b.ie13.B1850C5.f19_g16.iPI_dye.003_ANN_120001_124912_climo.nc"
HADCM3_PLIO_VW_PATH = HADCM3_SF_DIR / "MP_3.205Ma_xqbwb_V_and_W_3970_4000.nc"
HADCM3_PI_VW_PATH = HADCM3_SF_DIR / "PI_xqbwa_V_and_W_3900_4000.nc"
HADCM3_BASIN_PATH = HADCM3_SF_DIR / "basin_hadcm3"
CESM122_PLIO_PATH = CESM122_DIR / "Plio_ciso_T31_gx3v7_ALL.2901-3000_Reg1DegGrid.nc"
CESM122_PI_PATH = CESM122_DIR / "PreInd_ciso_T31_gx3v7_ALL.2901-3000_Reg1DegGrid.nc"

# Proxy and observational data
PLIOVAR_CHECKED_DIR = DATA_DIR / "PlioVAR_benthic" / "CHECKED"
PLIOVAR_SITE_DEPTHS_PATH = DATA_DIR / "PlioVAR_benthic" / "Site Depths.csv"
PLIOVAR_SST_PATH = DATA_DIR / "PlioVAR_SST" / "PlioVAR-KM5c_SST.tab"
SCHMITTNER_DIR = DATA_DIR / "Schmittner2017"
GEBCO_PATH = DATA_DIR / "GEBCO" / "GEBCO_2024_sub_ice_topo.nc"
WOA23_TEMP_PATH = DATA_DIR / "WOA23" / "woa23_decav_t00mn01.csv"
MODERN_D13C_PATH = DATA_DIR / "modern_d13C" / "C13_Climatology.nc"
MODERN_D18O_PATH = DATA_DIR / "modern_d18O" / "calculated_d18O_v1_1.nc"

# Export only the constants, so `from paths import *` never shadows names such
# as matplotlib.path.Path in the importing module.
__all__ = [name for name in dir() if name.isupper()]
