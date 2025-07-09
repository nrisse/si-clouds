"""
Creates latitude, longitude, and time boundary of HALO flights.
"""

import numpy as np
import pandas as pd
from lizard.ac3airlib import get_all_flights, meta
from lizard.readers.gps_ins import read_gps_ins


def main():
    """
    Prints the box for each HALO research flight
    """

    flights = get_all_flights(mission=["HALO-AC3"], platform="HALO")

    # loop over all flights
    for flight_id in flights:

        date, h0, h1, lon_min, lon_max, lat_min, lat_max = make_box(flight_id)

        info = (
            f"{flight_id}, "
            f"{date}, "
            f"{h0}, "
            f"{h1}, "
            f"{lon_min}, "
            f"{lon_max}, "
            f"{lat_min}, "
            f"{lat_max}"
        )

        print(info)


def get_all_boxes():
    """
    Returns all information needed to download ERA-5 for the specific flight.
    """

    flights = get_all_flights(mission=["HALO-AC3"], platform="HALO")

    boxes = {}

    # loop over all flights
    for flight_id in flights:

        date, h0, h1, lon_min, lon_max, lat_min, lat_max = make_box(flight_id)

        box = {
            "flight_id": flight_id,
            "date": date,
            "h0": h0,
            "h1": h1,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        }

        boxes[flight_id] = box

    return boxes


def make_box(flight_id):
    """
    Make lat/lon/time box around the flight
    """

    flight = meta(flight_id)

    ds_gps = read_gps_ins(flight_id)
    ds_gps = ds_gps.sel(time=slice(flight["takeoff"], flight["landing"]))

    h0 = flight["takeoff"]
    h1 = flight["landing"]
    date = flight["date"]

    lon_min = ds_gps.lon.min().item()
    lon_max = ds_gps.lon.max().item()
    lat_min = ds_gps.lat.min().item()
    lat_max = ds_gps.lat.max().item()

    # round lon_min to neasrest 0.25
    lon_min = np.round(lon_min * 4) / 4
    lon_max = np.round(lon_max * 4) / 4
    lat_min = np.round(lat_min * 4) / 4
    lat_max = np.round(lat_max * 4) / 4

    # round time to nearest hour
    h0 = pd.to_datetime(h0).round("h").hour
    h1 = pd.to_datetime(h1).round("h").hour

    return date, h0, h1, lon_min, lon_max, lat_min, lat_max


if __name__ == "__main__":
    main()
