"""
Calculates pressure, geopotential, and geopotential height at each ERA5 model
level. This code is based on ECMWF documentation and adapted to xarray and
netcdf input.

The code is fast enough to be executed online without saving additional files.

Required ERA5 variables:
- temperature (t)
- specific humidity (q)
- logarithm of surface pressure (lnsp)
- geopotential at surface (z)

Other requirements:
- ERA5 a and b coefficients at half model levels for pressure calculation

Online resources:
Equations: https://www.ecmwf.int/sites/default/files/elibrary/2023/81369-ifs-documentation-cy48r1-part-iii-dynamics-and-numerical-procedures.pdf
Code interim: https://confluence.ecmwf.int/display/CKB/ERA-Interim%3A+compute+geopotential+on+model+levels
Code ERA5: https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Pressureonmodellevels
"""

import numpy as np
import xarray as xr

from si_clouds.io.readers.era5 import read_era5_coef

R_D = 287.06
R_G = 9.80665


def model_to_pressure_height(ds_tq):
    """
    Calculates pressure, geopotential, and geoportential height at each model
    level.

    Parameters
    ----------
    ds_tq : xr.Dataset
        ERA5 model levels dataset with t and q at full levels.

    Returns
    -------
    ds_tq : xr.Dataset
        ERA5 model levels dataset with p, z, and h at full levels.
    """

    ds_coef = read_era5_coef()

    ds_tq, da_p_hl = calculate_pressure(ds_tq, ds_coef)
    ds_tq = calculate_geopotential(ds_tq, da_p_hl, ds_coef)

    assert ds_tq.h.sel(level=1).max() < 85000
    assert ds_tq.h.sel(level=1).min() > 70000

    return ds_tq


def calculate_pressure(ds_tq, ds_coef):
    """
    Calculates pressure at each model level.

    The pressure at the uppermost half level is set to 0.1 Pa instead of the
    0 Pa calculated from a and b coefficients.

    Parameters
    ----------
    ds_tq : xr.Dataset
        ERA5 model levels dataset with t and q at full levels.
    ds_coef : xr.Dataset
        ERA5 model level coefficients at half levels.

    Returns
    -------
    ds_tq : xr.Dataset
        ERA5 model levels dataset with p at full levels.
    da_p_hl : xr.DataArray
        Pressure at half levels.
    """

    # calculate pressure at half levels
    da_p_hl = ds_coef["a"] + ds_coef["b"] * np.exp(ds_tq["lnsp"])
    da_p_hl.loc[{"half_level": 0}] = 0.1

    # calculate pressure at model levels
    ds_tq["p"] = (
        da_p_hl.sel(half_level=ds_coef.ix_upper_level)
        + da_p_hl.sel(half_level=ds_coef.ix_lower_level)
    ) / 2

    return ds_tq, da_p_hl


def calculate_geopotential(ds_tq, da_p_hl, ds_coef):
    """
    Calculate geopotential at each model level.

    Parameters
    ----------
    ds_tq : xr.Dataset
        ERA5 model levels dataset with t and q at full levels.
    da_p_hl : xr.DataArray
        Pressure at half levels.
    ds_coef : xr.Dataset
        ERA5 model level coefficients at half levels and indices that map from
        level to half level (to not confuse which one lies above or below).
    """

    # calculate downward gradient of pressure and log pressure
    dp = da_p_hl.diff("half_level")
    dp = dp.rename({"half_level": "level"})

    dlog_p = np.log(da_p_hl).diff("half_level")
    dlog_p = dlog_p.rename({"half_level": "level"})

    # calculate alpha
    alpha = 1 - ((da_p_hl.sel(half_level=ds_coef.ix_upper_level) / dp) * dlog_p)
    alpha.loc[{"level": 1}] = np.log(2)  # at uppermost level
    alpha = alpha.reset_coords(drop=True)

    # calculate virtual potential temperature
    ds_tq["tv"] = ds_tq["t"] * (1 + 0.609133 * ds_tq["q"])

    # calculate geopotential at half levels
    da_z_hl = xr.zeros_like(da_p_hl)
    da_z_hl.loc[{"half_level": 137}] = ds_tq["z"]  # surface
    da_z_hl_above_sfc = ds_tq["z"] + (ds_tq["tv"] * R_D * dlog_p).sel(
        level=ds_tq["tv"].level[::-1]
    ).cumsum("level")
    da_z_hl_above_sfc = da_z_hl_above_sfc.rename({"level": "half_level"})
    da_z_hl_above_sfc["half_level"] = da_z_hl_above_sfc["half_level"] - 1  # move upward
    da_z_hl.loc[{"half_level": da_z_hl_above_sfc.half_level}] = da_z_hl_above_sfc

    # calculate geopotential and geopotential height at full level
    ds_tq["z"] = da_z_hl.sel(half_level=ds_coef.ix_lower_level) + (
        alpha * R_D * ds_tq["tv"]
    )
    ds_tq["h"] = ds_tq["z"] / R_G

    return ds_tq
