"""Generates a simple one target scenario

Using stonesoup's functions, a simple one target scenario is generated. The target is randomly ...
TODO
"""
import numpy as np
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

start_time = datetime.now()

def function():
    """

    :return:
    """
    return 0