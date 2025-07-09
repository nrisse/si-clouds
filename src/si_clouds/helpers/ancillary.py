"""
Prepares ancillary data for the retrieval.
"""

import os

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xhistogram.xarray as xhist
from lizard import ac3airlib
from lizard.ac3airlib import flights_of_day, get_all_flights, meta
from lizard.readers.amsr2_sic_track import read_amsr2_sic_track
from lizard.readers.gps_ins import read_gps_ins
from lizard.readers.halo_kt19 import read_halo_kt19
from lizard.readers.hamp import read_hamp
from lizard.readers.mira import read_all
from lizard.readers.wales import read_wales_dask
from lizard.writers.figure_to_file import write_figure

from si_clouds.helpers.distances import distance_to_coast, distance_to_sic
from si_clouds.helpers.ice_sondes import sea_ice_sondes
from si_clouds.io.readers.era5 import read_era5_single_levels_inst

THRESHOLD_ZE_MIN_HF = -40  # minimum ze for hydrometeor fraction
THRESHOLD_OFFSHORE = 15000  # distance to coast in m (offshore)
THRESHOLD_SIC = 90  # sea ice concentration (sea ice detection)
THRESHOLD_T2M = 273.15 - 1  # 2m temperature threshold
THRESHOLD_ZE = 5  # maximum radar reflectivity threshold
THRESHOLD_CS_HF = 0  # radar hydrometeor fraction clear-sky
THRESHOLD_BSR_MAX = 4  # backscatter ratio clear-sky
THRESHOLD_SONDE = 30  # sonde distance in seconds
THRESHOLD_ICE_EDGE = 200  # distance to ice edge in km
THRESHOLD_HF5KM = 0.5  # radar hydrometeor fraction from 0 to 5 km altitude
LIQUID_THRESHOLD_BSR = 50  # backscatter ratio threshold for liquid detection
LIQUID_THRESHOLD_DEP = 0.1  # depolarisation ratio threshold for liquid detection
THRESHOLD_ANGLE = 6  # roll and pitch angle threshold for filtering
ZE_MAX_HMIN, ZE_MAX_HMAX = 120, 10000
WALES_HMIN, WALES_HMAX = 50, 10000

# ensures that the times in text files are always shuffled the same way
RANDOM_SEED = 38
np.random.seed(RANDOM_SEED)


def main():

    flight_ids = get_all_flights("HALO-AC3", "HALO")

    ds_gps_ins = get_gps_data(flight_ids)
    ds = get_tb_data(flight_ids)
    ds_sic = get_sea_ice_data(flight_ids)
    ds_kt19 = get_kt19_data(flight_ids)
    ds_bsr, ds_dep = get_wales_data()
    ds_era = get_era5_alongtrack_data(ds, flight_ids)
    ds_mira = get_hamp_radar_data()
    ds_dsd = sea_ice_sondes()

    # roll and pitch filter
    ds = roll_pitch_filter(ds_gps_ins, ds)

    # now join with all datasets
    _, ds_gps_ins = xr.align(ds, ds_gps_ins, join="left")
    _, ds_era = xr.align(ds, ds_era, join="left")
    _, ds_kt19 = xr.align(ds, ds_kt19, join="left")
    _, ds_sic = xr.align(ds, ds_sic, join="left")
    _, ds_bsr = xr.align(ds, ds_bsr, join="left")
    _, ds_dep = xr.align(ds, ds_dep, join="left")
    _, ds_mira = xr.align(ds, ds_mira, join="left")

    # check wales statistics for the threshold definition
    plot_wales_histogram(ds_dep, ds_bsr)

    ds_anc = make_ancillary_dataset(
        ds, ds_gps_ins, ds_era, ds_sic, ds_kt19, ds_bsr, ds_dep, ds_mira, ds_dsd
    )

    write_times(ds, ds_anc)
    print_statistics(ds_anc)

    # total flown distance of the final data set
    print(ds_gps_ins["vel"].max().item())
    print(ds_gps_ins["vel"].min().item())
    print(ds_gps_ins["vel"].isnull().sum().item())
    print(
        "Flown distance in km:",
        ds_gps_ins["vel"]
        .sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid)
        .sum("time")
        .item()
        * 1e-3,
    )


def print_statistics(ds_anc):

    print(ds_anc.ix_sea_ice.sum().values)
    print(ds_anc.ix_clear_sky.sum().values)
    print(ds_anc.ix_clear_sky_kt19.sum().values)
    print(ds_anc.ix_clear_sky.sum().values / ds_anc.ix_sea_ice.sum().values)
    print(ds_anc.ix_clear_sky_kt19.sum().values / ds_anc.ix_sea_ice.sum().values)

    # fraction of data where the 2m temperature is high
    print(
        "Retrieval not valid due to T2M",
        (ds_anc.ix_sea_ice & ~ds_anc.ix_t2m_cold).sum().values
        / ds_anc.ix_sea_ice.sum().values,
    )
    print(
        "Retrieval not valid due to high Ze",
        (ds_anc.ix_sea_ice & ~ds_anc.ix_ze_low).sum().values
        / ds_anc.ix_sea_ice.sum().values,
    )
    print(
        "Retrieval not valid due to both",
        (ds_anc.ix_sea_ice & ~ds_anc.ix_retrieval_valid).sum().values
        / ds_anc.ix_sea_ice.sum().values,
    )

    # where the retrieval is valid
    print(
        ds_anc.time.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid)
        .count()
        .values
    )
    print(
        ds_anc.time.sel(time=ds_anc.ix_clear_sky & ds_anc.ix_retrieval_valid)
        .count()
        .values
    )
    print(
        ds_anc.time.sel(time=ds_anc.ix_clear_sky_kt19 & ds_anc.ix_retrieval_valid)
        .count()
        .values
    )
    print(
        ds_anc.time.sel(time=ds_anc.ix_clear_sky & ds_anc.ix_retrieval_valid)
        .count()
        .values
        / ds_anc.time.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid)
        .count()
        .values
    )
    print(
        ds_anc.time.sel(time=ds_anc.ix_clear_sky_kt19 & ds_anc.ix_retrieval_valid)
        .count()
        .values
        / ds_anc.time.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid)
        .count()
        .values
    )

    # location
    print(
        "Lon min",
        ds_anc.lon.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid).min().item(),
    )
    print(
        "Lon max",
        ds_anc.lon.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid).max().item(),
    )
    print(
        "Lat min",
        ds_anc.lat.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid).min().item(),
    )
    print(
        "Lat max",
        ds_anc.lat.sel(time=ds_anc.ix_sea_ice & ds_anc.ix_retrieval_valid).max().item(),
    )


def write_times(ds, ds_anc):
    """
    Write time and flight id of valid HAMP observations to text files for the
    retrieval runs. The times are shuffled to allow running a subset of the data
    for testing.
    """

    sets = [
        ("clear_sky_kt19", ds_anc.ix_clear_sky_kt19.values, True),
        ("all_sky", ds_anc.ix_sea_ice.values, True),
    ]

    for name, ix_set, shuffle in sets:

        times = ds.time.sel(time=ix_set).values

        if shuffle:
            np.random.shuffle(times)

        flight_ids = []
        for time in times:
            flight_ids.extend(
                list(
                    flights_of_day(
                        pd.Timestamp(time).date(),
                        missions=["HALO-AC3"],
                        platforms=["HALO"],
                    )
                )
            )

        print(times[0], times[-1])
        print(len(times), len(flight_ids))

        with open(
            os.path.join(
                os.environ["PATH_SEC"],
                f"data/sea_ice_clouds/oem_input/{name}.txt",
            ),
            "w",
        ) as f:
            for time, flight_id in zip(times, flight_ids):
                f.write(f"{time},{flight_id}\n")

    cases = [
        ("HALO-AC3_HALO_RF04", "HALO-AC3_HALO_RF04_hl07"),
        ("HALO-AC3_HALO_RF04", "HALO-AC3_HALO_RF04_hl08"),
        ("HALO-AC3_HALO_RF10", "HALO-AC3_HALO_RF10_ci01"),
        ("HALO-AC3_HALO_RF06", "HALO-AC3_HALO_RF06_hl12"),
        ("HALO-AC3_HALO_RF03", "HALO-AC3_HALO_RF03_hl09"),
        ("HALO-AC3_HALO_RF18", "HALO-AC3_HALO_RF18_hl03"),
        ("HALO-AC3_HALO_RF02", "HALO-AC3_HALO_RF02_hl11"),
        ("HALO-AC3_HALO_RF04", "HALO-AC3_HALO_RF04_hl11"),
    ]

    for flight_id, segment_id in cases:
        segment = ac3airlib.segments_dict(flight_id)[segment_id]
        t0, t1 = segment["start"], segment["end"]
        times = (
            ds.time.sel(time=ds_anc.ix_sea_ice.values).sel(time=slice(t0, t1)).values
        )
        flight_ids = []
        for time in times:
            flight_ids.extend(
                list(
                    flights_of_day(
                        pd.Timestamp(time).date(),
                        missions=["HALO-AC3"],
                        platforms=["HALO"],
                    )
                )
            )

        print(times[0], times[-1])
        print(len(times), len(flight_ids))

        with open(
            os.path.join(
                os.environ["PATH_SEC"],
                f"data/sea_ice_clouds/oem_input/{segment_id}.txt",
            ),
            "w",
        ) as f:
            for time, flight_id in zip(times, flight_ids):
                f.write(f"{time},{flight_id}\n")


def make_ancillary_dataset(
    ds, ds_gps_ins, ds_era, ds_sic, ds_kt19, ds_bsr, ds_dep, ds_mira, ds_dsd
):
    
    print("Creating ancillary dataset...")

    # availability of ancillary data
    ix_wales_valid = (ds_bsr["flags"] == 0).any("altitude").compute() & (
        ds_dep["flags"] == 0
    ).any("altitude").compute()
    ix_sic_valid = ~ds_sic.sic.isnull()
    ix_kt19_valid = ~ds_kt19.BT.isnull()

    # surface type
    ix_offshore = ds_gps_ins.dist_coast > THRESHOLD_OFFSHORE
    ix_sea_ice = ((ds_sic.sic > THRESHOLD_SIC) | (ds_gps_ins.lat > 88)) & ix_offshore
    ix_ocean = ds_sic.sic == 0
    ix_central_arctic = ds_gps_ins.dist_sic_0_50 > (THRESHOLD_ICE_EDGE * 1e3)

    # retrieval is valid
    ix_t2m_cold = ds_era.t2m < THRESHOLD_T2M  # 2m temperature threshold
    ix_ze_low = (ds_mira["dBZg_max"] < THRESHOLD_ZE) | (
        np.isnan(ds_mira["dBZg_max"])
    )  # maximum ze threshold
    ix_retrieval_valid = ix_t2m_cold & ix_ze_low & ix_sea_ice

    # high hydrometeor fraction
    ix_high_hf = ds_mira["hf_5km"] > THRESHOLD_HF5KM

    # liquid detection
    ix_liquid_region = (
        (ds_bsr.backscatter_ratio > LIQUID_THRESHOLD_BSR)
        & (ds_dep.aerosol_depolarisation < LIQUID_THRESHOLD_DEP)
    ).compute()
    ix_liquid = ix_liquid_region.any("altitude")
    liquid_top_height = (
        ix_liquid_region.cumsum("altitude").idxmax("altitude").where(ix_liquid)
    )

    # sea ice and radar+lidar clear-sky
    ix_clear_sky = (
        (ds_mira["hf_14km"] == THRESHOLD_CS_HF)
        & (ds_bsr["bsr_max"] < THRESHOLD_BSR_MAX)
        & ix_wales_valid
        & ix_sea_ice
    )
    ix_clear_sky_kt19 = ix_clear_sky & ix_kt19_valid

    # near dropsonde
    ix_sonde = np.abs((ds.time - ds_dsd.launch_time)).min("sonde_id") < np.timedelta64(
        THRESHOLD_SONDE, "s"
    )

    # clear-sky dropsonde
    ix_sonde_clear_sky = ix_sonde & ix_clear_sky
    ix_sonde_clear_sky_kt19 = ix_sonde & ix_clear_sky_kt19

    # cloud types (only when retrieval is valid)
    ix_all_liquid_0 = (
        ix_liquid
        & (liquid_top_height >= 0)
        & (liquid_top_height < 500)
        & ix_retrieval_valid
    )
    ix_all_liquid_1 = (
        ix_liquid
        & (liquid_top_height >= 500)
        & (liquid_top_height < 1000)
        & ix_retrieval_valid
    )
    ix_all_liquid_2 = (
        ix_liquid
        & (liquid_top_height >= 1000)
        & (liquid_top_height < 5000)
        & ix_retrieval_valid
    )
    ix_all_no_liquid = ~ix_liquid & ~ix_high_hf & ix_wales_valid & ix_retrieval_valid
    ix_all_potential_liquid = (
        (~ix_liquid & ix_high_hf | ix_liquid & (liquid_top_height >= 5000))
        & ix_wales_valid
        & ix_retrieval_valid
    )
    ix_all_nodata = ~ix_wales_valid & ix_retrieval_valid

    # cloud types in central arctic (only when retrieval is valid)
    ix_cea_liquid_0 = ix_central_arctic & ix_all_liquid_0
    ix_cea_liquid_1 = ix_central_arctic & ix_all_liquid_1
    ix_cea_liquid_2 = ix_central_arctic & ix_all_liquid_2
    ix_cea_no_liquid = ix_central_arctic & ix_all_no_liquid
    ix_cea_potential_liquid = ix_central_arctic & ix_all_potential_liquid
    ix_cea_nodata = ix_central_arctic & ix_all_nodata

    # cloud type consistency check
    assert (
        ix_all_liquid_0.sum()
        + ix_all_liquid_1.sum()
        + ix_all_liquid_2.sum()
        + ix_all_no_liquid.sum()
        + ix_all_potential_liquid.sum()
        + ix_all_nodata.sum()
    ) == ix_retrieval_valid.sum()

    # create one dataset that contains all the indices and other ancillary data
    ds_anc = xr.Dataset()
    ds_anc.coords["time"] = ds.time

    gps_vars = ["lat", "lon", "alt", "vel", "roll", "pitch", "heading"]
    for v in gps_vars:
        ds_anc[v] = ds_gps_ins[v]

    era5_vars = list(ds_era.data_vars)
    for v in era5_vars:
        ds_anc["era5_" + v] = ds_era[v]

    ds_anc["sic"] = ds_sic.sic

    ds_anc["kt19_bt"] = ds_kt19.BT
    ds_anc["bsr_max"] = ds_bsr["bsr_max"]
    ds_anc["bsr_altmax"] = ds_bsr["bsr_altmax"]
    ds_anc["hf_14km"] = ds_mira["hf_14km"]
    ds_anc["hf_5km"] = ds_mira["hf_5km"]
    ds_anc["liquid_top_height"] = liquid_top_height
    ds_anc["dBZg_max"] = ds_mira["dBZg_max"]

    # distances
    ds_anc["dist_coast"] = ds_gps_ins.dist_coast
    ds_anc["dist_sic_0_0"] = ds_gps_ins.dist_sic_0_0
    ds_anc["dist_sic_0_50"] = ds_gps_ins.dist_sic_0_50
    ds_anc["dist_sic_0_90"] = ds_gps_ins.dist_sic_0_90

    # thresholds
    ds_anc["threshold_sic"] = THRESHOLD_SIC
    ds_anc["threshold_t2m"] = THRESHOLD_T2M
    ds_anc["threshold_ze"] = THRESHOLD_ZE
    ds_anc["threshold_cs_hf"] = THRESHOLD_CS_HF
    ds_anc["threshold_bsr_max"] = THRESHOLD_BSR_MAX
    ds_anc["threshold_sonde"] = THRESHOLD_SONDE
    ds_anc["threshold_ice_edge"] = THRESHOLD_ICE_EDGE
    ds_anc["threshold_hf5km"] = THRESHOLD_HF5KM

    # indices
    ds_anc["ix_wales_valid"] = ix_wales_valid
    ds_anc["ix_sic_valid"] = ix_sic_valid
    ds_anc["ix_kt19_valid"] = ix_kt19_valid

    ds_anc["ix_offshore"] = ix_offshore
    ds_anc["ix_sea_ice"] = ix_sea_ice
    ds_anc["ix_ocean"] = ix_ocean
    ds_anc["ix_central_arctic"] = ix_central_arctic

    ds_anc["ix_t2m_cold"] = ix_t2m_cold
    ds_anc["ix_ze_low"] = ix_ze_low
    ds_anc["ix_retrieval_valid"] = ix_retrieval_valid

    ds_anc["ix_high_hf"] = ix_high_hf
    ds_anc["ix_liquid"] = ix_liquid

    ds_anc["ix_clear_sky"] = ix_clear_sky
    ds_anc["ix_clear_sky_kt19"] = ix_clear_sky_kt19
    ds_anc["ix_sonde"] = ix_sonde
    ds_anc["ix_sonde_clear_sky"] = ix_sonde_clear_sky
    ds_anc["ix_sonde_clear_sky_kt19"] = ix_sonde_clear_sky_kt19

    # cloud types
    ds_anc["ix_all_liquid_0"] = ix_all_liquid_0
    ds_anc["ix_all_liquid_1"] = ix_all_liquid_1
    ds_anc["ix_all_liquid_2"] = ix_all_liquid_2
    ds_anc["ix_all_no_liquid"] = ix_all_no_liquid
    ds_anc["ix_all_potential_liquid"] = ix_all_potential_liquid
    ds_anc["ix_all_nodata"] = ix_all_nodata

    # cloud types in central arctic
    ds_anc["ix_cea_liquid_0"] = ix_cea_liquid_0
    ds_anc["ix_cea_liquid_1"] = ix_cea_liquid_1
    ds_anc["ix_cea_liquid_2"] = ix_cea_liquid_2
    ds_anc["ix_cea_no_liquid"] = ix_cea_no_liquid
    ds_anc["ix_cea_potential_liquid"] = ix_cea_potential_liquid
    ds_anc["ix_cea_nodata"] = ix_cea_nodata

    # this is needed for era5 to avoid issues when reading the data
    for v in ds_anc.data_vars:
        ds_anc[v].encoding = {}

    print("Writing ancillary data to netCDF...")
    ds_anc.to_netcdf(
        os.path.join(
            os.environ["PATH_SEC"],
            f"data/sea_ice_clouds/detect/ancillary_data/ancillary.nc",
        )
    )
    print("Done writing ancillary data.")

    return ds_anc


def plot_wales_histogram(ds_dep, ds_bsr):

    da_hist_wales = xhist.histogram(
        ds_dep.aerosol_depolarisation.sel(altitude=slice(WALES_HMIN, WALES_HMAX)).chunk(
            {"time": 10000}
        ),
        np.log10(
            ds_bsr.backscatter_ratio.sel(altitude=slice(WALES_HMIN, WALES_HMAX)).chunk(
                {"time": 10000}
            )
        ),
        bins=[np.linspace(0, 1, 100), np.linspace(0, 6, 100)],
        dim=["time", "altitude"],
    ).compute()
    da_hist_wales = da_hist_wales.where(da_hist_wales > 0)

    fig, ax = plt.subplots(
        1, 1, figsize=(7, 5), sharex=True, sharey=True, layout="constrained"
    )

    im = ax.pcolormesh(
        da_hist_wales.aerosol_depolarisation_bin,
        da_hist_wales.backscatter_ratio_bin,
        np.log10(da_hist_wales).T,
        cmap=cmc.batlow,
    )
    fig.colorbar(im, ax=ax, label="Count [log$_{10}$]")

    ax.set_ylim(0, 4)
    ax.set_xlim(0, 0.7)

    ax.plot(
        [LIQUID_THRESHOLD_DEP, LIQUID_THRESHOLD_DEP],
        [np.log10(LIQUID_THRESHOLD_BSR), np.log10(1000000)],
        color="k",
        linewidth=2,
    )
    ax.plot(
        [0, LIQUID_THRESHOLD_DEP],
        [np.log10(LIQUID_THRESHOLD_BSR), np.log10(LIQUID_THRESHOLD_BSR)],
        color="k",
        linewidth=2,
    )

    ax.set_xlabel("Depolarization ratio")
    ax.set_ylabel("Backscatter ratio [log$_{10}$]")

    write_figure(fig, "wales_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_gps_data(flight_ids):
    ds_gps_ins_lst = []
    for flight_id in flight_ids:
        ds_gps_ins_lst.append(
            read_gps_ins(flight_id)
            .drop_vars(["sza", "saa"])
            .sel(time=slice(meta(flight_id)["takeoff"], meta(flight_id)["landing"]))
        )
    ds_gps_ins = xr.concat(ds_gps_ins_lst, dim="time")

    # compute distance to coastline
    ds_gps_ins["dist_coast"] = (
        "time",
        distance_to_coast(
            track_lon=ds_gps_ins.lon.values, track_lat=ds_gps_ins.lat.values
        ),
    )

    # compute distance to open water and low sea ice concentration areas
    ds_gps_ins["dist_sic_0_0"] = (
        "time",
        distance_to_sic(
            track_lon=ds_gps_ins.lon.values,
            track_lat=ds_gps_ins.lat.values,
            time=ds_gps_ins.time.values,
            sic_min=0,
            sic_max=0,
        ),
    )
    ds_gps_ins["dist_sic_0_50"] = (
        "time",
        distance_to_sic(
            track_lon=ds_gps_ins.lon.values,
            track_lat=ds_gps_ins.lat.values,
            time=ds_gps_ins.time.values,
            sic_min=0,
            sic_max=50,
        ),
    )
    ds_gps_ins["dist_sic_0_90"] = (
        "time",
        distance_to_sic(
            track_lon=ds_gps_ins.lon.values,
            track_lat=ds_gps_ins.lat.values,
            time=ds_gps_ins.time.values,
            sic_min=0,
            sic_max=90,
        ),
    )

    return ds_gps_ins


def get_tb_data(flight_ids):
    ds_lst = []
    for flight_id in flight_ids:
        ds_lst.append(
            read_hamp(flight_id).sel(
                time=slice(meta(flight_id)["takeoff"], meta(flight_id)["landing"])
            )
        )
    ds = xr.concat(ds_lst, dim="time")

    # drop where any channel is nan
    ds = ds.sel(time=~ds.tb.isnull().any("channel"))

    # here a drift occured at 118 GHz band
    t0 = np.datetime64("2022-04-12T12:49:00")
    t1 = np.datetime64("2022-04-12T13:02:00")
    ds = ds.sel(time=(ds.time < t0) | (ds.time > t1))

    return ds


def get_sea_ice_data(flight_ids):
    ds_sic_lst = []
    for flight_id in flight_ids:
        ds_sic_lst.append(read_amsr2_sic_track(flight_id))
    ds_sic = xr.concat(ds_sic_lst, dim="time")

    return ds_sic


def get_kt19_data(flight_ids):
    ds_lst = []
    for flight_id in flight_ids:
        try:
            ds_lst.append(
                read_halo_kt19(flight_id).sel(
                    time=slice(meta(flight_id)["takeoff"], meta(flight_id)["landing"])
                )
            )
        except FileNotFoundError:
            pass
    ds_kt19 = xr.concat(ds_lst, dim="time")

    return ds_kt19


def get_wales_data():
    ds_bsr = read_wales_dask(product="bsrgl", round_seconds=True)
    ds_dep = read_wales_dask(product="adepgl", round_seconds=True)

    # set to nan where data is flagged
    ds_bsr["backscatter_ratio"] = ds_bsr["backscatter_ratio"].where(
        ds_bsr["flags"] == 0
    )
    ds_dep["aerosol_depolarisation"] = ds_dep["aerosol_depolarisation"].where(
        ds_dep["flags"] == 0
    )

    # drop times where entire sample is nan
    da_bst_nan = ds_bsr.backscatter_ratio.isnull().all("altitude").compute()
    da_dep_nan = ds_dep.aerosol_depolarisation.isnull().all("altitude").compute()

    ds_bsr = ds_bsr.sel(time=~da_bst_nan)
    ds_dep = ds_dep.sel(time=~da_dep_nan)

    # align both wales datasets
    ds_dep, ds_bsr = xr.align(ds_dep, ds_bsr, join="inner")

    # compute backscatter ratio maximum and altitude of maximum
    ds_bsr["bsr_max"] = (
        ds_bsr.backscatter_ratio.sel(altitude=slice(WALES_HMIN, WALES_HMAX))
        .max("altitude")
        .compute()
    )
    ds_bsr["bsr_altmax"] = (
        ds_bsr.backscatter_ratio.sel(altitude=slice(WALES_HMIN, WALES_HMAX))
        .idxmax("altitude")
        .compute()
    )

    return ds_bsr, ds_dep


def get_era5_alongtrack_data(ds, flight_ids):
    ds_era_lst = []
    for flight_id in flight_ids:
        ds_era_flight = read_era5_single_levels_inst(flight_id)
        ds_hamp = ds.sel(
            time=slice(meta(flight_id)["takeoff"], meta(flight_id)["landing"])
        )
        ds_era_flight = ds_era_flight.sel(
            time=ds_hamp.time,
            longitude=ds_hamp.lon,
            latitude=ds_hamp.lat,
            method="nearest",
        )
        ds_era_lst.append(ds_era_flight)
    ds_era = xr.concat(ds_era_lst, dim="time")
    ds_era["time"] = ds.time
    ds_era = ds_era.reset_coords()

    # compute rainfall
    ds_era["rf"] = ds_era["tp"] - ds_era["sf"]

    return ds_era


def get_hamp_radar_data():
    ds_mira = read_all()
    ds_mira["hf"] = (ds_mira.dBZg > THRESHOLD_ZE_MIN_HF).compute()
    ds_mira["hf_14km"] = ds_mira["hf"].sel(height=slice(0, 14000)).mean("height")
    ds_mira["hf_5km"] = ds_mira["hf"].sel(height=slice(0, 5000)).mean("height")

    # get the maximum reflectivity (note that nan of maximum means no cloud)
    ds_mira["dBZg_max"] = (
        ds_mira.dBZg.sel(height=slice(ZE_MAX_HMIN, ZE_MAX_HMAX)).max("height").compute()
    )

    return ds_mira


def roll_pitch_filter(ds_gps_ins, ds):
    _, ds_gps_ins = xr.align(ds, ds_gps_ins, join="left")
    ix_angles = (np.abs(ds_gps_ins["roll"]) < THRESHOLD_ANGLE) & (
        np.abs(ds_gps_ins["pitch"]) < THRESHOLD_ANGLE
    )
    ds = ds.sel(time=ix_angles)

    return ds


if __name__ == "__main__":
    main()
