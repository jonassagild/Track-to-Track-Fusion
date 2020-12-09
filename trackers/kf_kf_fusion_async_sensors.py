"""kf_kf_fusion_unsync_sensors

Kalman filter fusion of two unsync sensors.
"""
from datetime import timedelta

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


class KFFusionUnsyncSensors:
    """
    todo
    """
    def __init__(self, start_time, prior: GaussianState, sigma_process=0.01,
                 sigma_meas_radar=3, sigma_meas_ais=1):
        """

        :param start_time:
        :param prior:
        :param sigma_process:
        :param sigma_meas_radar:
        :param sigma_meas_ais:
        """
        # start time
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

    def track(self, measurements_radar, measurements_ais, estimation_rate=1):
        """
        Uses the Kalman Filter to fuse the measurements received. Produces a new estimate at each estimation_rate.
        A prediction is performed when no new measurements are received when a new estimate is calculated.

        Note: when estimation_rate is lower than either of the measurements rates, it might not use all measurements
        when updating.

        :param measurements_radar:
        :param measurements_ais:
        :param estimation_rate: How often a new estimate should be calculated.
        """
        time = self.start_time
        tracks_fused = Track()
        tracks_radar = Track()

        # copy measurements
        measurements_radar = measurements_radar.copy()
        measurements_ais = measurements_ais.copy()
        # loop until there are no more measurements
        while measurements_ais or measurements_radar:
            # get all new measurements
            new_measurements_radar = \
                [measurement for measurement in measurements_radar if measurement.timestamp <= time]
            new_measurements_ais = \
                [measurement for measurement in measurements_ais if measurement.timestamp <= time]

            # remove the new measurements from the measurements lists
            for new_meas in new_measurements_ais:
                measurements_ais.remove(new_meas)
            for new_meas in new_measurements_radar:
                measurements_radar.remove(new_meas)

            # sort the new measurements
            new_measurements_radar.sort(key=lambda meas: meas.timestamp, reverse=True)
            new_measurements_ais.sort(key=lambda meas: meas.timestamp, reverse=True)

            while new_measurements_radar or new_measurements_ais:
                if new_measurements_radar and \
                        (not new_measurements_ais or
                         new_measurements_radar[0].timestamp <= new_measurements_ais[0].timestamp):
                    # predict and update with radar measurement
                    new_measurement = new_measurements_radar[0]
                    prediction = self.predictor.predict(self.prior, timestamp=new_measurement.timestamp)
                    hypothesis = SingleHypothesis(prediction, new_measurement)
                    post = self.updater_radar.update(hypothesis)
                    tracks_radar.append(post)
                    # remove measurement
                    new_measurements_radar.remove(new_measurement)
                else:
                    # predict and update with radar measurement
                    new_measurement = new_measurements_ais[0]
                    prediction = self.predictor.predict(self.prior, timestamp=new_measurement.timestamp)
                    hypothesis = SingleHypothesis(prediction, new_measurement)
                    post = self.updater_ais.update(hypothesis)
                    # remove measurement
                    new_measurements_ais.remove(new_measurement)

                # add to fused list
                self.prior = post

            # perform a prediction up until this time (the newest measurement might not be at this exact time)
            # note that this "prediction" might be the updated posterior, if the newest measurement was at this time
            prediction = self.predictor.predict(self.prior, timestamp=time)
            tracks_fused.append(GaussianState(prediction.mean, prediction.covar, prediction.timestamp))

            # increment time
            time += timedelta(seconds=estimation_rate)

        return tracks_fused, tracks_radar

    def plot(self, ground_truth, measurements_radar, measurements_ais, tracks_fused, tracks_radar):
        """
        Is this being used?

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
            w, v = np.linalg.eig(
                self.measurement_model_radar.matrix() @ state.covar @ self.measurement_model_radar.matrix().T)
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
            w, v = np.linalg.eig(
                self.measurement_model_radar.matrix() @ state.covar @ self.measurement_model_radar.matrix().T)
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


