""" kalman_filter_dependent_fusion

Uses the test and using taking the dependence into account. Follows Bar-Shaloms formulas for doing so.
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

from trackers.calc_cross_cov_estimate_error import calc_cross_cov_estimate_error


class kalman_filter_dependent_fusion:
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

        # same transition models (radar uses same as original)
        self.transition_model_radar = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(sigma_process_radar), ConstantVelocity(sigma_process_radar)])
        self.transition_model_ais = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(sigma_process_ais), ConstantVelocity(sigma_process_ais)])

        # same measurement models as used when generating the measurements
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
        """
        todo
        :return:
        """
        # create list for storing kalman gains
        kf_gains_radar = []
        kf_gains_ais = []

        # create list for storing transition_noise_covar
        transition_covars_radar = []
        transition_covars_ais = []

        # create list for storing tranisition matrixes
        transition_matrixes_radar = []
        transition_matrixes_ais = []

        # create list for storing tracks
        tracks_radar = Track()
        tracks_ais = Track()

        # track
        for measurement in self.measurements_radar:
            prediction = self.predictor_radar.predict(self.prior_radar, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)
            # calculate the kalman gain
            hypothesis.measurement_prediction = self.updater_radar.predict_measurement(hypothesis.prediction,
                                                                                       measurement_model=self.measurement_model_radar)
            post_cov, kalman_gain = self.updater_radar._posterior_covariance(hypothesis)
            kf_gains_radar.append(kalman_gain)
            # get the transition model covar
            predict_over_interval = measurement.timestamp - self.prior_radar.timestamp
            transition_covars_ais.append(self.transition_model_ais.covar(time_interval=predict_over_interval))
            transition_matrixes_ais.append(self.transition_model_ais.matrix(time_interval=predict_over_interval))
            # update
            post = self.updater_radar.update(hypothesis)
            tracks_radar.append(post)
            self.prior_radar = post

        for measurement in self.measurements_ais:
            prediction = self.predictor_radar.predict(self.prior_ais, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)
            # calculate the kalman gain
            hypothesis.measurement_prediction = self.updater_ais.predict_measurement(hypothesis.prediction,
                                                                                     measurement_model=self.measurement_model_ais)
            post_cov, kalman_gain = self.updater_ais._posterior_covariance(hypothesis)
            kf_gains_ais.append(kalman_gain)
            # get the transition model covar
            predict_over_interval = measurement.timestamp - self.prior_radar.timestamp
            transition_covars_radar.append(self.transition_model_radar.covar(time_interval=predict_over_interval))
            transition_matrixes_radar.append(self.transition_model_radar.matrix(time_interval=predict_over_interval))
            # update
            post = self.updater_ais.update(hypothesis)
            tracks_ais.append(post)
            self.prior_ais = post

        # FOR NOW: run track_to_track_association here, todo change pipeline flow
        # FOR NOW: run the association only when both have a new posterior (so each time the AIS has a posterior)
        # todo handle fusion when one track predicts and the other updates. (or both predicts) (Can't be done with the theory
        #  described in the article)

        cross_cov_ij = [np.zeros([4, 4])]
        cross_cov_ji = [np.zeros([4, 4])]

        # TODO change flow to assume that the indexes decide whether its from the same iterations
        # use indexes to loop through tracks, kf_gains etc

        tracks_fused = []
        # tracks_fused.append(tracks_radar[0])
        for i in range(1, len(tracks_radar)):
            # we assume that the indexes correlates with the timestamps. I.e. that the lists are 'synchronized'
            # check to make sure
            if tracks_ais[i].timestamp == tracks_radar[i].timestamp:
                # calculate the cross-covariance estimation error
                cross_cov_ji.append(calc_cross_cov_estimate_error(
                    self.measurement_model_ais.matrix(), self.measurement_model_radar.matrix(), kf_gains_ais[i],
                    kf_gains_radar[i],
                    transition_matrixes_ais[i], transition_covars_ais[i], cross_cov_ji[i - 1]
                ))
                cross_cov_ij.append(calc_cross_cov_estimate_error(
                    self.measurement_model_radar.matrix(), self.measurement_model_ais.matrix(), kf_gains_radar[i],
                    kf_gains_ais[i],
                    transition_matrixes_radar[i], transition_covars_ais[i], cross_cov_ij[i - 1]
                ))

                # test for track association
                # same_target = track_to_track_association.test_association_dependent_tracks(tracks_radar[i],
                #                                                                            tracks_ais[i],
                #                                                                            cross_cov_ij[i],
                #                                                                            cross_cov_ji[i], 0.01)
                same_target = True  # ignore test for track association for now
                if same_target:
                    fused_posterior, fused_covar = track_to_track_fusion.fuse_dependent_tracks(tracks_radar[i],
                                                                                               tracks_ais[i],
                                                                                               cross_cov_ij[i],
                                                                                               cross_cov_ji[i])
                    estimate = GaussianState(fused_posterior, fused_covar, timestamp=tracks_ais[i].timestamp)
                    tracks_fused.append(estimate)
        return tracks_fused, tracks_ais, tracks_radar

# # plot
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
# ax.set_title("Kalman filter tracking and fusion accounting for the dependence")
# fig.show()
# # fig.savefig("../results/scenario2/KF_tracking_and_fusion_accounting_for_dependence.svg")
#
# # plot estimate for estimate
# # plot
# fig_2 = plt.figure(figsize=(10, 6))
# ax = fig_2.add_subplot(1, 1, 1)
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.axis('equal')
# ax.plot([state.state_vector[0] for state in ground_truth],
#         [state.state_vector[2] for state in ground_truth],
#         linestyle="--",
#         label='Ground truth')
# # ax.scatter([state.state_vector[0] for state in measurements_radar],
# #            [state.state_vector[1] for state in measurements_radar],
# #            color='b',
# #            label='Measurements Radar')
# # ax.scatter([state.state_vector[0] for state in measurements_ais],
# #            [state.state_vector[1] for state in measurements_ais],
# #            color='r',
# #            label='Measurements AIS')
#
# for i in range(0, len(tracks_fused)):
#     # plot measurements
#     ax.scatter([measurements_radar[i + 1].state_vector[0]],
#                [measurements_radar[i + 1].state_vector[1]],
#                color='b',
#                label='Measurements Radar')
#     ax.scatter([measurements_ais[i + 1].state_vector[0]],
#                [measurements_ais[i + 1].state_vector[1]],
#                color='r',
#                label='Measurements AIS')
#
#     # plot one and one estimate
#     state_radar = tracks_radar[i + 1]
#     w, v = np.linalg.eig(measurement_model_radar.matrix() @ state_radar.covar @ measurement_model_radar.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_radar.state_vector[0], state_radar.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='b')
#     ax.add_artist(ellipse)
#
#     state_ais = tracks_ais[i + 1]
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state_ais.covar @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_ais.state_vector[0], state_ais.state_vector[2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.2,
#                       color='r')
#     ax.add_patch(ellipse)
#
#     state_fused = tracks_fused[i]
#     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state_fused[1] @ measurement_model_ais.matrix().T)
#     max_ind = np.argmax(w)
#     min_ind = np.argmin(w)
#     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
#     ellipse = Ellipse(xy=(state_fused[0][0], state_fused[0][2]),
#                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
#                       angle=np.rad2deg(orient),
#                       alpha=0.5,
#                       color='green')
#     ax.add_patch(ellipse)
#
#     fig_2.show()
#     input("Press Enter to continue...")
#
# #
# # # add ellipses to the posteriors
# # for state in tracks_radar:
# #     w, v = np.linalg.eig(measurement_model_radar.matrix() @ state.covar @ measurement_model_radar.matrix().T)
# #     max_ind = np.argmax(w)
# #     min_ind = np.argmin(w)
# #     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
# #     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
# #                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
# #                       angle=np.rad2deg(orient),
# #                       alpha=0.2,
# #                       color='b')
# #     ax.add_artist(ellipse)
# #
# # for state in tracks_ais:
# #     w, v = np.linalg.eig(measurement_model_ais.matrix() @ state.covar @ measurement_model_ais.matrix().T)
# #     max_ind = np.argmax(w)
# #     min_ind = np.argmin(w)
# #     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
# #     ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
# #                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
# #                       angle=np.rad2deg(orient),
# #                       alpha=0.2,
# #                       color='r')
# #     ax.add_patch(ellipse)
# #
# # for track_fused in tracks_fused:
# #     w, v = np.linalg.eig(measurement_model_ais.matrix() @ track_fused[1] @ measurement_model_ais.matrix().T)
# #     max_ind = np.argmax(w)
# #     min_ind = np.argmin(w)
# #     orient = np.arctan2(v[1, max_ind], v[0, max_ind])
# #     ellipse = Ellipse(xy=(track_fused[0][0], track_fused[0][2]),
# #                       width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
# #                       angle=np.rad2deg(orient),
# #                       alpha=0.5,
# #                       color='green')
# #     ax.add_patch(ellipse)
#
# fig_2.show()
