"""
Sigmoid function for the sensitivity fit.
"""

import numpy as np


def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d
