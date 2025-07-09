"""
Reads SMRT layer file.
"""

import os

import yaml
from dotenv import load_dotenv

import si_clouds

load_dotenv()


def read_smrt_layers(filename):
    """
    Read the SMRT setup.
    """

    with open(
        os.path.join(
            si_clouds.__path__[0],
            f"retrieval/settings/{filename}",
        ),
        "r",
    ) as f:
        smrt_layers = yaml.safe_load(f)

    return smrt_layers
