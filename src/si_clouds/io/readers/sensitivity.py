"""
Read parameters of the CWP sensitivity curve as a function of distance to ice edge.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def read_sensitivity_params():

    file = os.path.join(
        os.environ["PATH_SEC"],
        "data/sea_ice_clouds/cwp_sensitivity",
        "cwp_sensitivity.npy",
    )

    with open(file, "rb") as f:
        popt = np.load(f)

    return popt
