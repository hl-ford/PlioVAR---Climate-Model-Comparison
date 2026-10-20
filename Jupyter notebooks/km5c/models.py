"""Lazy, cached loaders for the model output (one function per model)."""

from functools import cache
from types import SimpleNamespace
import netCDF4 as nc
import numpy as np

from .gridding import _regrid_curvilinear_common_mask
from .paths import CESM122_DIR
from .paths import HADCM3_ISOTOPE_DIR
from .paths import ICESM12_ISOTOPE_DIR
from .paths import ICESM13_ISOTOPE_DIR


@cache
def load_icesm12():
    """iCESM1.2-ITPCAS KM5c (Eoi400) and 0 ka d18O and temperature; preind regridded to 1x1."""
    path = ICESM12_ISOTOPE_DIR 
    d18O_Pliocene = "Delta18O_ISO_Eoi400_1x1.monmean.nc" 
    plio_file_path = f'{path}/{d18O_Pliocene}' 

    d18O_preind = "Delta18O_iso_0ka_1x1_monmean.nc"
    preind_file_path = f'{path}/{d18O_preind}'

    temp_plio = "TEMP_ISO_Eoi400.2601-2700.1x1.monmean.nc"
    temp_plio_file_path = f'{path}/{temp_plio}'

    temp_preind = "TEMP_iso_0ka.0201-0300.1x1.monmean.nc"
    temp_preind_file_path = f'{path}/{temp_preind}'

    Plio_d18O_nc = nc.Dataset(plio_file_path)
    Plio_lon = Plio_d18O_nc.variables["lon"][:]
    Plio_lat = Plio_d18O_nc.variables["lat"][:]
    Plio_depth = Plio_d18O_nc.variables["z_t"][:]/100 # Convert depth to meters
    Plio_d18O_time = Plio_d18O_nc.variables["Delta18O"][:]
    Plio_temp_nc = nc.Dataset(temp_plio_file_path)
    Plio_temp_time = Plio_temp_nc.variables["TEMP"][:]

    preind_d18O_nc = nc.Dataset(preind_file_path)
    preind_lon_native = preind_d18O_nc.variables["TLONG"][:]
    preind_lat_native = preind_d18O_nc.variables["TLAT"][:]
    preind_depth = np.array(preind_d18O_nc.variables["z_t"])/100 # Convert depth to meters
    preind_d18O_time = preind_d18O_nc.variables["Delta18O"][:]
    preind_temp_nc = nc.Dataset(temp_preind_file_path)
    preind_temp_time = preind_temp_nc.variables["TEMP"][:]
    preind_temp_depth = np.array(preind_temp_nc.variables["z_t"])/100
    if not np.array_equal(preind_depth, preind_temp_depth):
        raise ValueError("Preindustrial isotope and temperature depth levels differ.")
    preind_temp_lat = preind_temp_nc.variables["lat"][:]
    preind_temp_lon = preind_temp_nc.variables["lon"][:]

    Plio_d18O = Plio_d18O_time.mean(axis=0)
    Plio_temp = Plio_temp_time.mean(axis=0)
    preind_d18O_native = preind_d18O_time.mean(axis=0)
    preind_temp = preind_temp_time.mean(axis=0)
    # Despite the filenames describing both products as 1x1 degrees, the isotope
    # file uses a native curvilinear grid. The shared preind_* fields below use the
    # regular temperature grid; *_native retains the original isotope field/coords.
    preind_d18O, preind_temp = _regrid_curvilinear_common_mask(
        preind_d18O_native, preind_lat_native, preind_lon_native,
        preind_temp, preind_temp_lat, preind_temp_lon,
    )
    preind_lon, preind_lat = np.meshgrid(preind_temp_lon, preind_temp_lat)
    Plio_SST = Plio_temp[0,:,:]
    preind_SST = preind_temp[0,:,:]


    return SimpleNamespace(**{k: v for k, v in locals().items() if k.startswith(('Plio_', 'preind_')) and not k.endswith(('_nc', '_time', '_file_path'))})


@cache
def load_hadcm3():
    """HadCM3 3.205 Ma and PI d18O and in-situ temperature."""
    path = HADCM3_ISOTOPE_DIR

    d18O_Pliocene = "MP_3.205Ma_xqbwb_d18o_and_temp3900_4000.nc"
    plio_file_path = f'{path}/{d18O_Pliocene}'

    d18O_preind = "PI_xqbwa_d18o_and_temp3900_4000.nc"
    preind_file_path = f'{path}/{d18O_preind}'

    Plio_d18O_nc = nc.Dataset(plio_file_path)

    preind_d18O_nc = nc.Dataset(preind_file_path)

    hadcm3_Plio_lon = Plio_d18O_nc.variables["longitude"][:]
    hadcm3_Plio_lat = Plio_d18O_nc.variables["latitude"][:]
    hadcm3_Plio_depth = Plio_d18O_nc.variables["depth_1"][:] #meters to km
    Plio_d18O_time = Plio_d18O_nc.variables["d18o_adjusted__global_mean___0_3permil"][:]
    Plio_temp_time = Plio_d18O_nc.variables["insitu_T"][:]

    hadcm3_preind_lon = preind_d18O_nc.variables["longitude"][:]
    hadcm3_preind_lat = preind_d18O_nc.variables["latitude"][:]
    hadcm3_preind_depth = preind_d18O_nc.variables["depth_1"][:] #meters to km
    preind_d18O_time = preind_d18O_nc.variables["d18o_adjusted__global_mean__0permil"][:]
    preind_temp_time = preind_d18O_nc.variables["insitu_T"][:]


    hadcm3_Plio_d18O = Plio_d18O_time
    hadcm3_Plio_temp = Plio_temp_time
    hadcm3_preind_d18O = preind_d18O_time
    hadcm3_preind_temp = preind_temp_time
    hadcm3_Plio_SST = hadcm3_Plio_temp[0,:,:]
    hadcm3_preind_SST = hadcm3_preind_temp[0,:,:]


    return SimpleNamespace(**{k[len('hadcm3_'):]: v for k, v in locals().items() if k.startswith('hadcm3_')})


@cache
def load_icesm13():
    """iCESM1.3 PlioMIP2 and PI d18O and temperature."""
    path = ICESM13_ISOTOPE_DIR

    d18O_Pliocene = "b.ie12.B1850.f19_g16.PMIP4-pliomip2.icesm1.3_ann_050101_060012_climo_1x1deg.nc"
    plio_file_path = f'{path}/{d18O_Pliocene}'

    d18O_preind = "b.e13.Bi1850C5.f19_g16.0ka.itrace.branch.pop.h.0001-100.ann_clim_compress_new.nc"
    preind_file_path = f'{path}/{d18O_preind}'

    Plio_d18O_nc = nc.Dataset(plio_file_path)

    preind_d18O_nc = nc.Dataset(preind_file_path)

    icesm13_Plio_lon = Plio_d18O_nc.variables["lon"][:]
    icesm13_Plio_lat = Plio_d18O_nc.variables["lat"][:]
    icesm13_Plio_depth = Plio_d18O_nc.variables["z_t"][:]/100 #meters to km
    Plio_d18O_time = Plio_d18O_nc.variables["Delta18O"][:]
    Plio_temp_time = Plio_d18O_nc.variables["TEMP"][:]

    icesm13_preind_lon = preind_d18O_nc.variables["TLONG"][:]
    icesm13_preind_lat = preind_d18O_nc.variables["TLAT"][:]
    icesm13_preind_depth = preind_d18O_nc.variables["z_t"][:]/100 #meters to km
    preind_d18O_time = (preind_d18O_nc.variables["R18O"][:] - 1) * 1000
    preind_temp_time = preind_d18O_nc.variables["TEMP"][:]
    print(Plio_d18O_time.shape)


    icesm13_Plio_d18O = Plio_d18O_time[0,:,:,:]

    print(icesm13_Plio_d18O.shape)
    icesm13_Plio_temp = Plio_temp_time[0,:,:,:]
    icesm13_preind_d18O = preind_d18O_time[0,:,:,:]
    icesm13_preind_temp = preind_temp_time[0,:,:,:]
    icesm13_Plio_SST = icesm13_Plio_temp[0,:,:]
    icesm13_preind_SST = icesm13_preind_temp[0,:,:]


    return SimpleNamespace(**{k[len('icesm13_'):]: v for k, v in locals().items() if k.startswith('icesm13_')})


@cache
def load_cesm122():
    """CESM1.2.2.1 Pliocene and PI d13C(DIC), temperature and nutrients."""
    path = CESM122_DIR

    d13C_Pliocene = "Plio_ciso_T31_gx3v7_ALL.2901-3000_Reg1DegGrid.nc"
    plio_file_path = f'{path}/{d13C_Pliocene}'

    d13C_preind = "PreInd_ciso_T31_gx3v7_ALL.2901-3000_Reg1DegGrid.nc"
    preind_file_path = f'{path}/{d13C_preind}'

    Plio_d13C_nc = nc.Dataset(plio_file_path)

    preind_d13C_nc = nc.Dataset(preind_file_path)

    cesm122_Plio_lon = Plio_d13C_nc.variables["lon"][:]
    cesm122_Plio_lat = Plio_d13C_nc.variables["lat"][:]
    cesm122_Plio_depth = Plio_d13C_nc.variables["z_w"][:]/100 #meters to km
    Plio_d13C_time = Plio_d13C_nc.variables["CISO_DIC_d13C"][:]
    cesm122_Plio_temp = Plio_d13C_nc.variables["TEMP"][-1,:,:,:]
    cesm122_Plio_SST = cesm122_Plio_temp[0,:,:]

    cesm122_preind_lon = preind_d13C_nc.variables["lon"][:]
    cesm122_preind_lat = preind_d13C_nc.variables["lat"][:]
    cesm122_preind_depth = preind_d13C_nc.variables["z_w"][:]/100 #meters to km
    preind_d13C_time = preind_d13C_nc.variables["CISO_DIC_d13C"][:]
    cesm122_preind_PO4 = preind_d13C_nc.variables["PO4"][:]
    cesm122_preind_NO3 = preind_d13C_nc.variables["NO3"][:]
    cesm122_preind_temp = preind_d13C_nc.variables["TEMP"][:]
    cesm122_preind_SST = cesm122_preind_temp[0,:,:]


    cesm122_preind_NO3 = cesm122_preind_NO3[-1,:,:,:]

    cesm122_Plio_d13C = Plio_d13C_time[-1,:,:,:]
    cesm122_preind_d13C = preind_d13C_time[-1,:,:,:]
    cesm122_preind_PO4 = cesm122_preind_PO4[-1,:,:,:]

    return SimpleNamespace(**{k[len('cesm122_'):]: v for k, v in locals().items() if k.startswith('cesm122_')})
