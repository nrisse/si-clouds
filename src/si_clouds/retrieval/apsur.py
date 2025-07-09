"""
A priori parameters for surface.
"""

import numpy as np
from lizard.readers.halo_kt19 import read_halo_kt19

from si_clouds.io.readers.era5 import read_era5_single_levels_inst
from si_clouds.io.readers.smrt_layers import read_smrt_layers


def apriori_surface(
    flight_id,
    time,
    lat,
    lon,
    source_t_as,
    ir_emissivity,
    smrt_layers_filename,
    tas2tsi_factor,
):
    """
    Provides a priori values for surface parameters.

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    time : datetime
        Time of the observation.

    Returns
    -------
    dict
        A priori values for surface parameters.
    """

    if source_t_as == "era5":
        ts = apriori_ts_era5(flight_id, time, lat, lon)
    elif source_t_as == "kt19":
        ts = apriori_ts_kt19(flight_id, time, ir_emissivity=ir_emissivity)
    else:
        ts = np.nan

    layer_dct = apriori_layer(filename=smrt_layers_filename)
    surf_dct = {
        "t_as": ts,
        "t_si": tas2tsi(ts, factor=tas2tsi_factor),
        **layer_dct,
    }

    return surf_dct


def apriori_layer(filename):
    """
    A priori information of layer parameters. This is read from the SMRT layer
    definition file.

    Units from file and conversions:

    - corr_length: m -> mm
    - density: kg/m^3
    """

    smrt_layers = read_smrt_layers(filename)

    # unit conversion factors
    conversions = {"corr_length": 1000, "density": 1, "thickness": 1}

    layers = ["wind_slab", "depth_hoar"]
    parameters = ["corr_length", "density", "thickness"]

    layer_dct = {}

    for layer in layers:
        for parameter in parameters:

            values = smrt_layers["base_layers"]["snow"][layer][parameter]

            c = conversions[parameter]

            layer_dct[layer + "_" + parameter] = values[0] * c
            layer_dct[layer + "_" + parameter + "_std"] = values[1] * c

    return layer_dct


def apriori_ts_era5(flight_id, time, lat, lon):
    """
    A priori surface temperature. The source can be either ERA5 or KT-19 on
    board the aircraft.
    """

    ds = read_era5_single_levels_inst(flight_id)

    # select the closest grid point
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest", tolerance=0.25)

    # select the closest time
    ds = ds.sel(time=time, method="nearest")

    # apply linear correction, which was derived from kt-19 and era5 data
    ts = 0.9432970421020019 * ds.skt + 10.832421240419194

    return ts.item()


def apriori_ts_kt19(flight_id, time, ir_emissivity):

    ds_kt19 = read_halo_kt19(flight_id)

    # select closest time
    ds_kt19 = ds_kt19.sel(time=time, method="nearest")

    # compute ts from infrared emissivity and TB
    ts = ds_kt19.BT.values / ir_emissivity

    print("Time offset to KT-19: ", time - ds_kt19.time.values)

    return ts


def tas2tsi(tas, factor):
    """Air-snow interface temperature to snow-ice interface temperature."""

    t_water = 273.15 - 1.8
    t_diff = t_water - tas

    tsi = tas + t_diff * factor

    return tsi


def compute_tas2tsi_factor(tas, tsi):
    """
    Derive factor to get from air-snow interface temperature to snow-ice
    interface temperature.
    """

    t_water = 273.15 - 1.8
    t_diff = t_water - tas

    factor = (tsi - tas) / t_diff

    return factor
