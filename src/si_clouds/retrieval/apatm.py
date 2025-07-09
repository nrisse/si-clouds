"""
Create a priori atmospheric profiles. Two options are implemented:
- A priori from radiosonde climatology at Ny-Alesund.
- A priori from isntantaneous ERA5 model levels data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from lizard.readers.radiosonde import read_merged_radiosonde
from metpy.calc import (
    mixing_ratio_from_relative_humidity,
    specific_humidity_from_mixing_ratio,
)
from metpy.units import units

from si_clouds.helpers.ice_sondes import sea_ice_sondes
from si_clouds.helpers.model2pressure import model_to_pressure_height
from si_clouds.io.readers.dropsondes import read_dropsondes_l2
from si_clouds.io.readers.era5 import (
    read_era5_model_levels_clim,
    read_era5_model_levels_inst,
)


def profile_dropsonde_era5(flight_id, time, lat, lon):
    """
    Combination of ERA5 and dropsonde profile. Based on the flight, time, lat,
    and lon information, the ERA5 pixel is selected and combined with the
    temporally closest dropsonde launch of the same flight.

    Temperature and specific humidity are averaged over the ERA5 model levels
    based on the model level height. ERA5 values are replaced where dropsonde
    information is available. No interpolation of the dropsonde is performed
    in case of missing data.
    """

    # get era5 profile
    t, qv, p, h = apriori_mean_era5(flight_id, time, lat, lon)

    # get dropsonde profile
    ds_dsd = sea_ice_sondes(sic_threshold=80)
    ds_dsd = ds_dsd.swap_dims({"sonde_id": "launch_time"})
    ds_dsd = ds_dsd.sel(launch_time=time, method="nearest")
    print(
        "Dropsonde time:",
        ds_dsd.launch_time.values,
        "Dropsonde lat:",
        ds_dsd.flight_lat.values,
        "Dropsonde lon:",
        ds_dsd.flight_lon.values,
    )
    print(
        "Time difference between dropsonde and era5:",
        ds_dsd.launch_time.values - time,
    )

    # read corresponding level 2 sonde
    ds_l2 = read_dropsondes_l2(flight_id=flight_id, time=ds_dsd.launch_time.values)
    ds_l2["qv"] = specific_humidity_from_mixing_ratio(
        mixing_ratio=mixing_ratio_from_relative_humidity(
            pressure=ds_l2["p"] * units.Pa,
            temperature=ds_l2["ta"] * units.K,
            relative_humidity=ds_l2["rh"] * 100 * units.percent,
        )
    )
    ds_l2 = ds_l2.swap_dims({"time": "gpsalt"})
    ds_l2 = ds_l2.sel(gpsalt=ds_l2.gpsalt.notnull())
    ds_l2 = ds_l2.rename({"gpsalt": "h", "ta": "t"})
    df_l2 = ds_l2[["qv", "t"]].reset_coords(drop=True).to_dataframe()
    df_l2 = df_l2.dropna(axis=0, how="all")

    dh = h[1:] - h[:-1]
    h_bins = np.append(np.append(h[0] - dh[0] / 2, h[:-1] + dh / 2), h[-1] + dh[-1] / 2)
    ix = pd.cut(df_l2.index, bins=h_bins)
    df_l2 = df_l2.groupby(ix).mean()

    # convert specific humidity to logarithmic units log10(g/kg)
    df_l2["qv"] = np.log10(df_l2["qv"] * 1e3)

    # replace era5 value by dropsonde value where available
    t = np.where(df_l2["t"].notnull(), df_l2["t"], t)
    qv = np.where(df_l2["qv"].notnull(), df_l2["qv"], qv)

    return t, qv, p, h


def apriori_mean_era5(flight_id, time, lat, lon):
    """
    Loads a priori mean atmosphere from era5 for a given flight, location, and
    time.

    Parameters
    ----------
    flight_id : str
        Flight identifier.
    time : str
        Time of observation to get nearest a priori era5 profile
    lat : float
        Latitude of observation to get nearest a priori era5 profile
    lon : float
        Longitude of observation to get nearest a priori era5 profile

    Returns
    -------
    t : np.ndarray
        Temperature profile in K.
    qv : np.ndarray
        Specific humidity profile in log10(g/kg).
    p : np.ndarray
        Pressure profile in Pa.
    z : np.ndarray
        Geopotential height profile in m.
    """

    ds = read_era5_model_levels_inst(flight_id)

    # select the closest grid point
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest", tolerance=0.25)

    # select the closest time
    ds = ds.sel(time=time, method="nearest")

    print("Search time: ", time, "Search lat:", lat, "Search lon:", lon)
    print(
        "ERA5 time:",
        ds.time.values,
        "ERA5 lat:",
        ds.latitude.values,
        "ERA5 lon:",
        ds.longitude.values,
    )

    # adds pressure, geopotential, and geopotential height
    ds = model_to_pressure_height(ds)

    # convert specific humidity to logarithmic units log10(g/kg)
    ds["q"] = np.log10(ds["q"] * 1e3)

    # extract variables as numpy array
    ds = ds.sel(level=ds.level[::-1])  # first is lowest

    # thermodynamic variables
    t = ds.t.values
    qv = ds.q.values

    # pressure and geometric height
    p = ds.p.values
    h = ds.h.values

    return t, qv, p, h


def apriori_covariance_era5(n):
    """
    Loads a priori covariance matrix from era5. This is global for all flights.

    Parameters
    ----------
    n : int
        Number of retrieved atmospheric layers (from surface).
    """

    ds = read_era5_model_levels_clim()

    # reduce to number of layers
    ds = ds.sel(level=slice(137 - n + 1, 137))

    # drop cloud information
    ds = ds[["t", "q"]]

    # convert specific humidity to logarithmic units log10(g/kg)
    ds["q"] = np.log10(ds["q"] * 1e3)

    ds = ds.sel(level=ds.level[::-1])  # first is lowest

    # stack all profiles
    ds = ds.stack(i=["time", "latitude", "longitude"])  # now (level, i)

    x = np.concatenate(
        [ds[v].values for v in ["t", "q"]],
        axis=0,
    )

    cov = np.cov(x)

    return cov


def apriori_atmosphere(dz0, dz1, n, n_retrieved):
    """
    Creates atmospheric a priori states from radiosonde seasonal climatology.

    Parameters
    ----------
    dz0 : float
        Height of the first grid bin in meters.
    dz1 : float
        Height of the last grid bin in meters.
    n : int
        Number of grid bins.
    n_retrieved : int
        Number of retrieved atmospheric profiles counted from bottom to top.
        This ensures that the covariance matrix has the right dimensions for
        the retrieval and does not include height levels that are not
        retrieved.

    Returns
    -------
    ds_mean_retrieval : xarray.Dataset
        A priori mean atmospheric profiles.
    ds_cov_retrieval : xarray.Dataset
        A priori covariance matrix of atmospheric profiles.
    ds_mean : xarray.Dataset
        A priori mean atmospheric profiles including all height levels. This
        is the same as ds_mean_retrieval but with all height levels.
    """

    z_grid, z_bnds = create_height_grid(dz0, dz1, n)

    ds = radiosonde_climatology(z_grid, z_bnds)

    ds_retrieval = ds.isel(z_lev=slice(0, n_retrieved))

    ds_cov_retrieval = apriori_covariance(ds_retrieval)
    ds_mean = apriori_state(ds)

    ds_mean_retrieval = ds_mean.isel(z_lev=slice(0, n_retrieved))

    return ds_mean_retrieval, ds_cov_retrieval, ds_mean, z_bnds


def create_height_grid(dz0, dz1, n):
    """
    Create height grid of the retrieval.

    Parameters
    ----------
    dz0 : float
        Height of the first grid bin in meters.
    dz1 : float
        Height of the last grid bin in meters.
    n : int
        Number of grid bins.

    Returns
    -------
    z_grid : np.array
        Height grid bin center
    z_bnds : np.array
        Height grid boundaries
    """

    dz = np.linspace(dz0, dz1, n)
    z_bnds = np.append([0], dz.cumsum())
    z_grid = (z_bnds[1:] + z_bnds[:-1]) / 2

    return z_grid, z_bnds


def apriori_covariance(ds, x_vars=["temp", "qv"]):
    """
    Creates a priori covariance matrix. Only atmospheric T anq q profiles.
    """

    x_vars_long = []
    for var in x_vars:
        for i in range(len(ds.z_lev)):
            x_vars_long.append(f"{var}_{i}")

    cov_lst = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        # combine temperature and humidity into one dataset
        x = np.concatenate(
            [ds[v].sel(time=ds.season == season).values for v in x_vars],
            axis=0,
        )

        cov_lst.append(np.cov(x))

    da_cov = xr.DataArray(
        np.stack(cov_lst),
        dims=["season", "x", "x"],
        coords={"season": ["DJF", "MAM", "JJA", "SON"], "x": x_vars_long},
    )

    return da_cov


def apriori_state(ds):
    """
    Creates a priori state vector. Only atmospheric T and q profiles.
    """

    ds_mean = ds.groupby(ds.season).mean()

    return ds_mean


def radiosonde_climatology(z_grid, z_bnds):
    """
    Interpolates radiosondes onto retrieval grid for the calculation of the
    covariance matrix and the seasonal mean profiles.

    Units of output
    ---------------
    temp : K
    qv : log10(g/kg)
    press : Pa

    Parameters
    ----------
    z_grid : array-like
        Retrieval height grid in meters. The values are interpreted as center
        of the retrieval grid bins.
    z_bnds : array-like
        Boundary of the z grid bins.
    """

    ds_hres = read_merged_radiosonde()

    # seasonal information
    ds_hres = ds_hres.swap_dims({"i_sonde": "time"})

    # unit conversion
    ds_hres["temp"] = ds_hres["temp"] + 273.15
    ds_hres["press"] = ds_hres["press"] * 100

    # set relative humididity of 0 to 0.01
    ds_hres["rh"] = xr.where(ds_hres["rh"] > 0.01, ds_hres["rh"], 0.01)

    # calculate specific humidity
    ds_hres["qv"] = (
        specific_humidity_from_mixing_ratio(
            mixing_ratio=mixing_ratio_from_relative_humidity(
                pressure=ds_hres["press"] * units.Pa,
                temperature=ds_hres["temp"] * units.K,
                relative_humidity=ds_hres["rh"] * units.percent,
            )
        )
        * 1e3
    )

    # set negative humidity to nan
    ds_hres["qv"] = ds_hres["qv"].where(ds_hres["qv"] >= 0)

    # convert specific humidity to logarithmic units
    ds_hres["qv"] = np.log10(ds_hres["qv"])

    # replace -inf values with nans
    ds_hres["qv"] = ds_hres["qv"].where(ds_hres["qv"] != -np.inf)

    # interpolate sondes onto retrieval grid
    ds = (
        ds_hres[["temp", "qv", "press"]]
        .groupby_bins("z_lev", z_bnds, labels=z_grid)
        .mean()
    )
    ds = ds.rename({"z_lev_bins": "z_lev"})
    ds["season"] = ds_hres.time.dt.season

    return ds
