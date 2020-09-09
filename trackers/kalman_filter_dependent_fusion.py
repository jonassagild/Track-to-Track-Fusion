""" kalman_filter

TODO
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

from data_fusion import track_to_track_association
from data_fusion import track_to_track_fusion

# load ground truth and the measurements
ground_truth = open_object.open_object("../scenarios/scenario2/ground_truth.pk1")
measurements_radar = open_object.open_object("../scenarios/scenario2/measurements_radar.pk1")
measurements_ais = open_object.open_object("../scenarios/scenario2/measurements_ais.pk1")

# load start_time
start_time = open_object.open_object("../scenarios/scenario2/start_time.pk1")

# same transition models (radar uses same as original)
transition_model_radar = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01), ConstantVelocity(0.01)])
transition_model_ais = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01), ConstantVelocity(0.01)])

# same measurement models as used when generating the measurements
# Specify measurement model for radar
measurement_model_radar = LinearGaussian(
    ndim_state=4,  # number of state dimensions
    mapping=(0, 2),  # mapping measurement vector index to state index
    noise_covar=np.array([[1, 0],  # covariance matrix for Gaussian PDF
                          [0, 1]])
)

# Specify measurement model for AIS
measurement_model_ais = LinearGaussian(
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
updater_ais = KalmanUpdater(measurement_model_ais)

# create prior, both trackers use the same starting point
prior_radar = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
prior_ais = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

tracks_radar = Track()
for measurement in measurements_radar:
    prediction = predictor_radar.predict(prior_radar, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater_radar.update(hypothesis)
    tracks_radar.append(post)
    prior_radar = tracks_radar[-1]

tracks_ais = Track()
for measurement in measurements_ais:
    prediction = predictor_radar.predict(prior_ais, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater_ais.update(hypothesis)
    tracks_ais.append(post)
    prior_ais = tracks_ais[-1]

# FOR NOW: run track_to_track_association here, todo change pipeline flow
# FOR NOW: run the association only when both have a new posterior (so each time the AIS has a posterior)
# todo handle fusion when one track predicts and the other updates. (or both predicts) (Can't be done with the theory
#  described in the article)
tracks_fused = []
for track_ais in tracks_ais:
    # find a track in tracks_radar with the same timestamp
    for track_radar in tracks_radar:
        if track_ais.timestamp == track_radar.timestamp:
            same_target = track_to_track_association.test_association(track_radar, track_ais, 0.01)
            if same_target:
                fused_posterior, fused_covar = track_to_track_fusion.fuse(track_radar, track_ais)
                tracks_fused.append((fused_posterior, fused_covar))
            break

# plot
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

for state in tracks_ais:
    w, v = np.linalg.eig(measurement_model_ais.matrix() @ state.covar @ measurement_model_ais.matrix().T)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='r')
    ax.add_patch(ellipse)

for track_fused in tracks_fused:
    w, v = np.linalg.eig(measurement_model_ais.matrix() @ track_fused[1] @ measurement_model_ais.matrix().T)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
    ellipse = Ellipse(xy=(track_fused[0][0], track_fused[0][2]),
                      width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                      angle=np.rad2deg(orient),
                      alpha=0.5,
                      color='green')
    ax.add_patch(ellipse)

# add ellipses to add legend todo do this less ugly
ellipse = Ellipse(xy=(0, 0),
                  width=0,
                  height=0,
                  color='r',
                  alpha=0.2,
                  label='Posterior AIS')
ax.add_patch(ellipse)
ellipse = Ellipse(xy=(0, 0),
                  width=0,
                  height=0,
                  color='b',
                  alpha=0.2,
                  label='Posterior Radar')
ax.add_patch(ellipse)
ellipse = Ellipse(xy=(0, 0),
                  width=0,
                  height=0,
                  color='green',
                  alpha=0.5,
                  label='Posterior Fused')
ax.add_patch(ellipse)

ax.legend()
ax.set_title("Kalman filter tracking and fusion under the error independence assumption")
fig.show()
fig.savefig("../results/scenario1/KF_tracking_and_fusion_under_error_independence_assumption.svg")












