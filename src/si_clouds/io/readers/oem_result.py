"""
Read optimal estimation results from file.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def read_oem_result_concat(version, test_id, write=True):
    """
    Provides the option to read from or write to concatenated files.
    """

    path = os.path.join(os.environ["PATH_RET"])

    if write:
        ds_a, ds_op, ds_i, ds_syn = read_oem_result(
            flight_id="HALO-AC3_HALO*",
            version=version,
            test_id=test_id,
        )

        ds_a.to_netcdf(os.path.join(path, f"x_a_{version}{test_id}.nc"))
        ds_op.to_netcdf(os.path.join(path, f"x_op_{version}{test_id}.nc"))
        ds_i.to_netcdf(os.path.join(path, f"x_i_{version}{test_id}.nc"))
        if ds_syn is not None:
            ds_syn.to_netcdf(os.path.join(path, f"x_syn_{version}{test_id}.nc"))

    else:
        ds_a = xr.open_dataset(os.path.join(path, f"x_a_{version}{test_id}.nc"))
        ds_op = xr.open_dataset(os.path.join(path, f"x_op_{version}{test_id}.nc"))
        # in case iterations are not stored
        try:
            ds_i = xr.open_dataset(os.path.join(path, f"x_i_{version}{test_id}.nc"))
        except FileNotFoundError:
            ds_i = None
        try:
            ds_syn = xr.open_dataset(os.path.join(path, f"x_syn_{version}{test_id}.nc"))
        except FileNotFoundError:
            ds_syn = None

    return ds_a, ds_op, ds_i, ds_syn


def read_oem_result(flight_id, time=None, version="", test_id=""):
    """
    Read OEM result for a specific flight and time.
    """

    files_a, files_op, files_i, files_syn = get_oem_filenames(
        flight_id=flight_id, time=time, version=version, test_id=test_id
    )

    print("Reading a priori files")
    ds_a = read_files(files_a)

    print("Reading optimal solution files")
    ds_op = read_files(files_op)

    print("Reading iterations")
    ds_i = read_files(files_i)

    if test_id != "":
        print("Reading synthetic retrieval")
        ds_syn = read_files(files_syn)
    else:
        ds_syn = None

    # clean encodings
    datasets = [ds_a, ds_op, ds_i, ds_syn]
    for ds in datasets:
        if ds is None:
            continue
        for var in ds.data_vars:
            ds[var].encoding.clear()

    return ds_a, ds_op, ds_i, ds_syn


def read_files(files):
    if len(files) == 1:
        if os.path.exists(files[0]):
            ds = xr.open_dataset(files[0])
            ds = ds.expand_dims("time")
        else:
            ds = None
    elif len(files) > 1:
        ds = concat_along_time(files)
    else:
        ds = None
    return ds


def concat_along_time(files):
    ds_lst = []
    for file in tqdm(files):
        ds_single = xr.open_dataset(file).expand_dims("time")
        # add coordinates
        if "iteration" in ds_single.dims:
            ds_single.coords["iteration"] = np.arange(len(ds_single.iteration))
        ds_lst.append(ds_single)
    print("Concatenating along time")
    ds = xr.concat(ds_lst, dim="time")
    return ds


def get_oem_filenames(flight_id, time=None, version="", test_id=""):
    """
    Gets filenames of OEM output. Either for a specific time or for all times.
    """

    path = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/retrieved",
    )

    if time is not None:
        time = pd.Timestamp(time)
        filename = f"{flight_id}_{time.strftime('%Y%m%d_%H%M%S')}_{version}{test_id}.nc"
        return [
            [os.path.join(path, f"x_a_{filename}")],
            [os.path.join(path, f"x_op_{filename}")],
            [os.path.join(path, f"x_i_{filename}")],
            [os.path.join(path, f"x_syn_{filename}")],
        ]

    else:
        filename = f"{flight_id}_*_{version}{test_id}.nc"
        return [
            sorted(glob(os.path.join(path, f"x_a_{filename}"))),
            sorted(glob(os.path.join(path, f"x_op_{filename}"))),
            sorted(glob(os.path.join(path, f"x_i_{filename}"))),
            sorted(glob(os.path.join(path, f"x_syn_{filename}"))),
        ]
