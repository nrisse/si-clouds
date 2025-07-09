"""
Reads time and flight id with valid HAMP data.
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def read_retrieval_times(subset):
    """
    Select subset "all_sky", "clear_sky_kt19", ...
    """

    df = pd.read_csv(
        os.path.join(
            os.environ["PATH_SEC"], "data/sea_ice_clouds/oem_input", f"{subset}.txt"
        ),
        header=None,
    )

    return df
