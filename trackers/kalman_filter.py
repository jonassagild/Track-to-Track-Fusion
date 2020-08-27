""" kalman_filter

TODO
"""

import numpy as np
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater

from utils import open_object

# load ground truth and the measurements
ground_truth = open_object.open_object("../scenarios/scenario1/ground_truth.pk1")
measurements_radar = open_object.open_object("../scenarios/scenario1/measurements_radar.pk1")
measurements_ais = open_object.open_object("../scenarios/scenario1/measurements_AIS.pk1")

# same transition models (radar uses same as original)
transition_model_radar = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01), ConstantVelocity(0.01)])
transition_model_ais = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.02), ConstantVelocity(0.02)])

# same measurement models as used when generating the measurements
# Specify measurement model for radar
measurement_model_radar = LinearGaussian(
    ndim_state=4,  # number of state dimensions
    mapping=(0, 2),  # mapping measurement vector index to state index
    noise_covar=np.array([[3, 0],  # covariance matrix for Gaussian PDF
                          [0, 3]])
)

# Specify measurement model for AIS
measurement_model_AIS = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[1, 0],
                          [0, 1]])
)

# specify predictors
predictor_radar = KalmanPredictor(transition_model_radar)
predictor_ais = KalmanPredictor(transition_model_ais)

# specify updaters
updater_radar = KalmanUpdater(measurement_model_radar)
updater_ais = KalmanUpdater(measurements_ais)

# create prior, both use the same
prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)


# todo maybe move to another file

# todo: start with doing everything in one file, and then split the functionality?



