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

# create prior
prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

tracks_fused = Track()
tracks_radar = Track()
for measurement_idx in range(0, len(measurements_radar)):
    # radar measurement every timestep, AIS measurement every second
    # first predict, then update with radar measurement. Then every second iteration, perform an extra update step
    # using the AIS measurement
    measurement_radar = measurements_radar[measurement_idx]

    prediction = predictor.predict(prior, timestamp=measurement_radar.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement_radar)
    post = updater_radar.update(hypothesis)

    # save radar track
    tracks_radar.append(post)

    if measurement_idx % 2:
        measurement_ais = measurements_ais[measurement_idx//2]
        hypothesis = SingleHypothesis(post, measurement_ais)
        post = updater_ais.update(hypothesis)

    # save fused track
    tracks_fused.append(post)
    prior = tracks_fused[-1]

# PLOT
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')
ax.plot([state.state_vector[0] for state in ground_truth],
        [state.state_vector[2] for state in ground_truth],
        linestyle="--",
        label='Ground truth')
ax.scatter([state.state_vector[0] for state in measurements_radar],
           [state.state_vector[1] for state in measurements_radar],
           color='b',
           label='Measurements Radar')
ax.scatter([state.state_vector[0] for state in measurements_ais],
           [state.state_vector[1] for state in measurements_ais],
           color='r',
           label='Measurements AIS')

# add ellipses to the posteriors
for state in tracks_fused:
    w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='r')
    ax.add_artist(ellipse)

for state in tracks_radar:
    w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='b')
    ax.add_artist(ellipse)

# add ellipses to add legend todo do this less ugly
ellipse = Ellipse(xy=(0, 0),
                  width=0,
                  height=0,
                  color='r',
                  alpha=0.2,
                  label='Posterior Fused')
ax.add_patch(ellipse)

ellipse = Ellipse(xy=(0, 0),
                  width=0,
                  height=0,
                  color='b',
                  alpha=0.2,
                  label='Posterior Radar')
ax.add_patch(ellipse)

ax.legend()
ax.set_title("Kalman filter tracking and fusion when AIS is viewed as a measurement")
fig.show()
fig.savefig("../results/scenario1/KF_tracking_and_fusion_viewing_ais_as_measurement.svg")







