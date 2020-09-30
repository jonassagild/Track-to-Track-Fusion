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

class kalman_filter_independent_fusion:
    """
    todo
    """
    def __init__(self, measurements_radar, measurements_ais, start_time, prior: GaussianState,
                 sigma_process_radar=0.01, sigma_process_ais=0.01, sigma_meas_radar=3, sigma_meas_ais=1):
        """

        :param measurements_radar:
        :param measurements_ais:
        :param start_time:
        :param prior:
        :param sigma_process_radar:
        :param sigma_process_ais:
        :param sigma_meas_radar:
        :param sigma_meas_ais:
        """
        self.start_time = start_time
        self.measurements_radar = measurements_radar
        self.measurements_ais = measurements_ais

        # transition models (process models)
        self.transition_model_radar = CombinedLinearGaussianTransitionModel([ConstantVelocity(sigma_process_radar),
                                                                             ConstantVelocity(sigma_process_radar)])
        self.transition_model_ais = CombinedLinearGaussianTransitionModel([ConstantVelocity(sigma_process_ais),
                                                                           ConstantVelocity(sigma_process_ais)])

        # Specify measurement model for radar
        self.measurement_model_radar = LinearGaussian(
            ndim_state=4,  # number of state dimensions
            mapping=(0, 2),  # mapping measurement vector index to state index
            noise_covar=np.array([[sigma_meas_radar, 0],  # covariance matrix for Gaussian PDF
                                  [0, sigma_meas_radar]])
        )

        # Specify measurement model for AIS
        self.measurement_model_ais = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[sigma_meas_ais, 0],
                                  [0, sigma_meas_ais]])
        )

        # specify predictors
        self.predictor_radar = KalmanPredictor(self.transition_model_radar)
        self.predictor_ais = KalmanPredictor(self.transition_model_ais)

        # specify updaters
        self.updater_radar = KalmanUpdater(self.measurement_model_radar)
        self.updater_ais = KalmanUpdater(self.measurement_model_ais)

        # create prior, both trackers use the same starting point
        self.prior_radar = prior
        self.prior_ais = prior

    def track(self):
        self.tracks_radar = Track()
        for measurement in self.measurements_radar:
            prediction = self.predictor_radar.predict(self.prior_radar, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)
            post = self.updater_radar.update(hypothesis)
            self.tracks_radar.append(post)
            self.prior_radar = self.tracks_radar[-1]

        self.tracks_ais = Track()
        for measurement in self.measurements_ais:
            prediction = self.predictor_radar.predict(self.prior_ais, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)
            post = self.updater_ais.update(hypothesis)
            self.tracks_ais.append(post)
            self.prior_ais = self.tracks_ais[-1]

        self.tracks_fused = self._fuse_tracks(self.tracks_radar, self.tracks_ais)

        return self.tracks_fused, self.tracks_radar, self.tracks_ais

    def _fuse_tracks(self, tracks_radar, tracks_ais):
        tracks_fused = []
        for track_radar in tracks_radar:
            # find a track in tracks_radar with the same timestamp
            estimate = track_radar
            for track_ais in tracks_ais:
                if track_ais.timestamp == track_radar.timestamp:
                    # same_target = track_to_track_association.test_association_independent_tracks(track_radar, track_ais,
                    #                                                                              0.01)
                    same_target = True  # ignore association for now
                    if same_target:
                        fused_posterior, fused_covar = track_to_track_fusion.fuse_independent_tracks(track_radar,
                                                                                                     track_ais)
                        estimate = GaussianState(fused_posterior, fused_covar, timestamp=track_radar.timestamp)
                    break
            tracks_fused.append(estimate)
        return tracks_fused


# plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.axis('equal')
# ax.plot([state.state_vector[0] for state in ground_truth],
#         [state.state_vector[2] for state in ground_truth],
#         linestyle="--",
#         label='Ground truth')
# ax.scatter([state.state_vector[0] for state in measurements_radar],
#            [state.state_vector[1] for state in measurements_radar],
#            color='b',
#            label='Measurements Radar')
# ax.scatter([state.state_vector[0] for state in measurements_ais],
#            [state.state_vector[1] for state in measurements_ais],
#            color='r',
#            label='Measurements AIS')
#
# # add ellipses to the posteriors
# for state in tracks_radar:
#     w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='b')
#     ax.add_artist(ellipse)
#
# for state in tracks_ais:
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state.covar @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='r')
#     ax.add_patch(ellipse)
#
# for track_fused in tracks_fused:
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ track_fused[1] @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(track_fused[0][0], track_fused[0][2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.5,
#                       color='green')
#     ax.add_patch(ellipse)
#
# # add ellipses to add legend todo do this less ugly
# ellipse = Ellipse(xy=(0, 0),
#                   width=0,
#                   height=0,
#                   color='r',
#                   alpha=0.2,
#                   label='Posterior AIS')
# ax.add_patch(ellipse)
# ellipse = Ellipse(xy=(0, 0),
#                   width=0,
#                   height=0,
#                   color='b',
#                   alpha=0.2,
#                   label='Posterior Radar')
# ax.add_patch(ellipse)
# ellipse = Ellipse(xy=(0, 0),
#                   width=0,
#                   height=0,
#                   color='green',
#                   alpha=0.5,
#                   label='Posterior Fused')
# ax.add_patch(ellipse)
#
# ax.legend()
# ax.set_title("Kalman filter tracking and fusion under the error independence assumption")
# fig.show()
# fig.savefig("../results/scenario1/KF_tracking_and_fusion_under_error_independence_assumption.svg")












