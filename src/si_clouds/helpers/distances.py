"""
Compute distance to coast for a given lat lon set.
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
from lizard.readers.amsr2_sic import read_amsr2_sic
from scipy.spatial import KDTree


def distance_to_coast(track_lon, track_lat):
    """
    Compute distance to coast for a given lat lon set
    """

    projection = ccrs.NorthPolarStereo()
    wgs84 = ccrs.PlateCarree()

    # get coastline coordinates
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="physical", name="coastline"
    )
    coastlines = list(shpreader.Reader(shpfilename).geometries())

    coast_lon, coast_lat = [], []
    for geom in coastlines:
        coast_lon.extend(geom.xy[0])
        coast_lat.extend(geom.xy[1])
    coast_lon = np.array(coast_lon)
    coast_lat = np.array(coast_lat)
    coast_lon = coast_lon[coast_lat > 65]
    coast_lat = coast_lat[coast_lat > 65]

    # transform coordinates
    coast_proj = projection.transform_points(wgs84, coast_lon, coast_lat)
    track_proj = projection.transform_points(wgs84, track_lon, track_lat)

    # kdtree
    kdtree = KDTree(coast_proj[:, :2])
    dist, idx = kdtree.query(track_proj[:, :2])

    return dist


def distance_to_sic(track_lon, track_lat, time, sic_min=0, sic_max=0):
    """
    Compute distance to areas with low sea ice concentration. This is based
    on the daily AMSR2 sea ice concentration data. When using a threshold of
    0%, it gives an indication of the distance to the sea ice edge.
    """

    projection = ccrs.NorthPolarStereo()
    wgs84 = ccrs.PlateCarree()

    dates = pd.to_datetime(time).date
    distance = np.full(len(track_lon), np.nan)
    for date in np.unique(dates):
        ix = dates == date

        ds_sic = read_amsr2_sic(
            date=date,
            path=f"/data/obs/campaigns/halo-ac3/auxiliary/sea_ice/daily_grid/",
        )

        # get lon lat values where sic is smaller equal threshold
        ds_sic = ds_sic.stack({"xy": ["x", "y"]})
        ds_sic = ds_sic.sel(xy=(ds_sic.sic >= sic_min) & (ds_sic.sic <= sic_max))
        sic_lon = ds_sic.lon.values
        sic_lat = ds_sic.lat.values

        # transform coordinates
        coast_proj = projection.transform_points(wgs84, sic_lon, sic_lat)
        track_proj = projection.transform_points(wgs84, track_lon[ix], track_lat[ix])

        # kdtree
        kdtree = KDTree(coast_proj[:, :2])
        dist, idx = kdtree.query(track_proj[:, :2])

        distance[ix] = dist

    return distance
