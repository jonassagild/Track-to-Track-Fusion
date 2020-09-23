"""kalman_filter_view_AIS_as_measurement

Views the AIS measurement as a "pure" measurement. Uses the update step of the kalman filter to fuse the AIS and
Radar measurements.
"""
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from utils import open_object

# load ground truth and the measurements
ground_truth = open_object.open_object("../scenarios/scenario1/ground_truth.pk1")
measurements_radar = open_object.open_object("../scenarios/scenario1/measurements_radar.pk1")
measurements_ais = open_object.open_object("../scenarios/scenario1/measurements_ais.pk1")

# load start_time
start_time = open_object.open_object("../scenarios/scenario1/start_time.pk1")

# only one transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01), ConstantVelocity(0.01)])

# same measurement models as used when generating the measurements
# Specify measurement model for radar
measurement_model_radar = LinearGaussian(
    ndim_state=4,  # number of state dimensions
    mapping=(0, 2),  # mapping measurement vector index to state index
    noise_covar=np.array([[3, 0],  # covariance matrix for Gaussian PDF
                          [0, 3]])
)

# Specify measurement model for AIS
measurement_model_ais = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[1, 0],
                          [0, 1]])
)

# specify predictor
predictor = KalmanPredictor(transition_model)

# specify updaters
updater_radar = KalmanUpdater(measurement_model_radar)
updater_ais = KalmanUpdater(measurement_model_ais)



