"""kalman_filter_view_AIS_as_measurement

Views the AIS measurement as a "pure" measurement. Uses the update step of the kalman filter to fuse the AIS and
Radar measurements.
"""
import numpy as np
import scipy

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from scripts.generate_scenario import generate_scenario

from utils import open_object
from utils import calc_metrics


class kalman_filter_ais_as_measurement:
    """
    todo
    """
    def __init__(self, measurements_radar, measurements_ais, start_time, prior: GaussianState, sigma_process=0.01,
                 sigma_meas_radar=3,
                 sigma_meas_ais=1):
        """

        :param measurements_radar:
        :param measurements_ais:
        :param start_time:
        :param prior:
        :param sigma_process:
        :param sigma_meas_radar:
        :param sigma_meas_ais:
        """
        # measurements and start time
        self.measurements_radar = measurements_radar
        self.measurements_ais = measurements_ais
        self.start_time = start_time

        # transition model
        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(sigma_process),
                                                                       ConstantVelocity(sigma_process)])

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

        # specify predictor
        self.predictor = KalmanPredictor(self.transition_model)

        # specify updaters
        self.updater_radar = KalmanUpdater(self.measurement_model_radar)
        self.updater_ais = KalmanUpdater(self.measurement_model_ais)

        # create prior todo move later and probably rename
        self.prior = prior

    def track(self):
        self.tracks_fused = Track()
        self.tracks_radar = Track()
        for measurement_idx in range(0, len(self.measurements_radar)):
            # radar measurement every timestep, AIS measurement every second
            # first predict, then update with radar measurement. Then every second iteration, perform an extra update step
            # using the AIS measurement
            measurement_radar = self.measurements_radar[measurement_idx]

            prediction = self.predictor.predict(self.prior, timestamp=measurement_radar.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement_radar)
            post = self.updater_radar.update(hypothesis)

            # save radar track
            self.tracks_radar.append(post)

            if measurement_idx % 2:
                measurement_ais = self.measurements_ais[measurement_idx // 2]
                hypothesis = SingleHypothesis(post, measurement_ais)
                post = self.updater_ais.update(hypothesis)

            # save fused track
            self.tracks_fused.append(post)
            prior = self.tracks_fused[-1]
        return self.tracks_fused, self.tracks_radar

    def plot(self, ground_truth):
        """

        :return:
        """
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
        ax.scatter([state.state_vector[0] for state in self.measurements_radar],
                   [state.state_vector[1] for state in self.measurements_radar],
                   color='b',
                   label='Measurements Radar')
        ax.scatter([state.state_vector[0] for state in self.measurements_ais],
                   [state.state_vector[1] for state in self.measurements_ais],
                   color='r',
                   label='Measurements AIS')

        # add ellipses to the posteriors
        for state in self.tracks_fused:
            w, v = np.linalg.eig(self.measurement_model_radar.matrix() @ state.covar @ self.measurement_model_radar.matrix().T)
            max_ind = np.argmax(w)
            min_ind = np.argmin(w)
            orient = np.arctan2(v[1, max_ind], v[0, max_ind])
            ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                              width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                              angle=np.rad2deg(orient),
                              alpha=0.2,
                              color='r')
            ax.add_artist(ellipse)

        for state in self.tracks_radar:
            w, v = np.linalg.eig(self.measurement_model_radar.matrix() @ state.covar @ self.measurement_model_radar.matrix().T)
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

        # todo move or remove
        ax.legend()
        ax.set_title("Kalman filter tracking and fusion when AIS is viewed as a measurement")
        fig.show()
        fig.savefig("../results/scenario1/KF_tracking_and_fusion_viewing_ais_as_measurement.svg")


