"""
Get ERA5 data from climate date store (CDS) on model and single levels for
HALO-AC3. Climatological coarser resolution and instantaneous finer resolution
are supported. One file is saved per day.

How to run the script:
python era5download.py instantaneous model
python era5download.py climatology model 2017

https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
https://apps.ecmwf.int/codes/grib/param-db/
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
"""

import os
import sys
from datetime import datetime

import cdsapi
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from si_clouds.helpers.halobounds import get_all_boxes

load_dotenv()


FORMAT = "netcdf"
PARAMS = {
    "model": {
        130: "Temperature",
        133: "Specific humidity",
        247: "Specific cloud ice water content",
        246: "Specific cloud liquid water content",
        75: "Specific rain water content",
        76: "Specific snow water content",
        152: "Logarithm of surface pressure",
        129: "Geopotential",
    },
    "single": {
        "instant": [
            "2m_temperature",
            "high_cloud_cover",
            "ice_temperature_layer_1",
            "ice_temperature_layer_2",
            "ice_temperature_layer_3",
            "ice_temperature_layer_4",
            "land_sea_mask",
            "low_cloud_cover",
            "mean_sea_level_pressure",
            "medium_cloud_cover",
            "sea_ice_cover",
            "sea_surface_temperature",
            "skin_temperature",
            "surface_pressure",
            "total_cloud_cover",
            "total_column_cloud_ice_water",
            "total_column_cloud_liquid_water",
            "total_column_rain_water",
            "total_column_snow_water",
            "total_column_supercooled_liquid_water",
            "total_column_water_vapour",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "precipitation_type",
            "2m_dewpoint_temperature",
        ],
        "accum": [
            "total_precipitation",
            "snowfall",
        ],
    },
}


def main(kind, product):
    """
    Download ERA-5.

    Parameters
    ----------
    kind : str
        'climatology' or 'instantaneous'
    product : str
        'model' or 'single' or 'model_single'
    """

    print(f"{datetime.now()}\nDownloading ERA-5 {kind} on {product}")

    if kind == "climatology":

        download_climatology(product=product)

    elif kind == "instantaneous":

        download_instantaneous(product=product)


def download_climatology(product):
    """
    Download climatological region
    """

    # properties of climatological data
    dates = pd.date_range("2017-01-01", "2021-12-31", freq="1D")
    dates = dates[(dates.month == 3) | (dates.month == 4)]
    area = [82, 10, 82, 10]  # n/w/s/e
    grid = "0.25/0.25"
    time = "00/to/23/by/6"
    h0 = 0
    h1 = 23
    hstep = 6

    for date in dates:

        # define path where retrieved data is stored
        path = os.path.join(
            os.environ["PATH_SEC"],
            "data/era5/si_clouds",
            date.strftime("%Y/%m/"),
        )
        if not os.path.exists(path):
            os.makedirs(path)

        if "model" in product:
            file_ml = os.path.join(
                path,
                f'era5-model-levels_siclouds_{date.strftime("%Y%m%d")}.nc',
            )
            download_era5_model_levels(
                date=date, time=time, grid=grid, area=area, file_target=file_ml
            )
            print(f"Written file: {file_ml}")

        if "single" in product:
            file_sl = os.path.join(
                path,
                f'era5-single-levels_siclouds_{date.strftime("%Y%m%d")}.nc',
            )
            download_era5_single_levels(
                date=date,
                area=area,
                hour0=h0,
                hour1=h1,
                hstep=hstep,
                file_target=file_sl,
            )
            print(f"Written file: {file_sl}")


def download_instantaneous(product):
    """
    Download hourly data
    """

    boxes = get_all_boxes()

    grid = "0.25/0.25"
    hstep = 1

    if "model" in product:
        for flight_id, box in boxes.items():

            print(flight_id)

            date = box["date"]
            area = [
                box["lat_max"],
                box["lon_min"],
                box["lat_min"],
                box["lon_max"],
            ]  # n/w/s/e
            time = f"{str(box['h0']).zfill(2)}/to/{str(box['h1']).zfill(2)}/by/1"

            # define path where retrieved data is stored
            path = os.path.join(
                os.environ["PATH_SEC"],
                "data/era5/si_clouds_flights",
                date.strftime("%Y/%m/"),
            )
            if not os.path.exists(path):
                os.makedirs(path)
            file_ml = os.path.join(
                path,
                f'era5-model-levels_siclouds_{date.strftime("%Y%m%d")}.nc',
            )
            download_era5_model_levels(
                date=date, time=time, grid=grid, area=area, file_target=file_ml
            )
            print(f"Written file: {file_ml}")

    if "single" in product:
        # for the entire campaign, get the lat lon bounds
        # and download data with additional +/-5/30 days
        lon_min = np.min([boxes[f]["lon_min"] for f in boxes])
        lat_min = np.min([boxes[f]["lat_min"] for f in boxes])
        lon_max = np.max([boxes[f]["lon_max"] for f in boxes])
        lat_max = np.max([boxes[f]["lat_max"] for f in boxes])
        date_start = np.min([boxes[f]["date"] for f in boxes])
        date_end = np.max([boxes[f]["date"] for f in boxes])
        dates = pd.date_range(
            date_start - pd.Timedelta(30, "d"),
            date_end + pd.Timedelta(5, "d"),
            freq="1D",
        )

        area = [
            lat_max,
            lon_min,
            lat_min,
            lon_max,
        ]  # n/w/s/e

        for date in dates:
            print(date)

            # define path where retrieved data is stored
            path = os.path.join(
                os.environ["PATH_SEC"],
                "data/era5/si_clouds_flights",
                date.strftime("%Y/%m/"),
            )
            if not os.path.exists(path):
                os.makedirs(path)

            file_sl = os.path.join(
                path,
                f'era5-single-levels_siclouds_{date.strftime("%Y%m%d")}.nc',
            )
            download_era5_single_levels(
                date=date,
                area=area,
                hour0=0,
                hour1=23,
                hstep=hstep,
                file_target=file_sl,
            )
            print(f"Written file: {file_sl}")


def download_era5_model_levels(date, time, grid, area, file_target):
    """
    Downloads ERA5 on model levels for pre-defined variables
    """

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-complete",
        {
            "date": str(date),
            "levelist": "1/to/137",
            "levtype": "ml",
            "param": "/".join([str(i) for i in PARAMS["model"].keys()]),
            "stream": "oper",
            "time": time,
            "type": "an",
            "area": "/".join([str(i) for i in area]),
            "grid": grid,
            "format": FORMAT,
        },
        file_target,
    )


def download_era5_single_levels(date, area, hour0, hour1, hstep, file_target):
    """
    Downloads ERA5 on single levels for pre-defined variables
    """
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "year": [str(date.year)],
        "month": [str(date.month).zfill(2)],
        "day": [str(date.day).zfill(2)],
        "time": [f"{hour:02d}:00" for hour in range(hour0, hour1 + 1, hstep)],
        "data_format": FORMAT,
        "download_format": "unarchived",
        "area": area,
    }

    for variable_type in PARAMS["single"]:
        request["variable"] = PARAMS["single"][variable_type]
        client = cdsapi.Client()
        client.retrieve(
            dataset,
            request,
            file_target.replace(".nc", f"_{variable_type}.nc"),
        )


if __name__ == "__main__":

    kind = sys.argv[1]
    product = sys.argv[2]

    print(f"Requesting ERA-5 {product}")

    main(kind=kind, product=product)
