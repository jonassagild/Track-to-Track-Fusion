"""Generates a simple one target scenario

Using stonesoup's functions, a simple one target scenario is generated. The target is randomly ...
TODO

Inspiration for how to generate ground truth and measurements are taken from
https://stonesoup.readthedocs.io/en/latest/auto_tutorials/01_KalmanFilterTutorial.html#sphx-glr-auto-tutorials-01-kalmanfiltertutorial-py
"""

from datetime import datetime
from datetime import timedelta
import numpy as np

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

from utils import store_object


def generate_scenario(seed=1996, permanent_save=True, sigma_transition=0.01, sigma_meas_radar=3, sigma_meas_ais=1):
    # specify seed to be able repeat example
    start_time = datetime.now()

    np.random.seed(seed)

    # combine two 1-D CV models to create a 2-D CV model
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(sigma_transition),
                                                              ConstantVelocity(sigma_transition)])

    # starting at 0,0 and moving NE
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

    # generate truth using transition_model and noise
    for k in range(1, 21):
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)))

    # Simulate measurements
    # Specify measurement model for radar
    measurement_model_radar = LinearGaussian(
        ndim_state=4,  # number of state dimensions
        mapping=(0, 2),  # mapping measurement vector index to state index
        noise_covar=np.array([[sigma_meas_radar, 0],  # covariance matrix for Gaussian PDF
                              [0, sigma_meas_radar]])
    )

    # Specify measurement model for AIS
    measurement_model_ais = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[sigma_meas_ais, 0],
                              [0, sigma_meas_ais]])
    )

    # generate "radar" measurements
    measurements_radar = []
    for state in truth:
        measurement = measurement_model_radar.function(state, noise=True)
        measurements_radar.append(Detection(measurement, timestamp=state.timestamp))

    # generate "AIS" measurements
    measurements_ais = []
    state_num = 0
    for state in truth:
        state_num += 1
        if not state_num % 2:  # measurement every second time step
            measurement = measurement_model_ais.function(state, noise=True)
            measurements_ais.append(Detection(measurement, timestamp=state.timestamp))

    if permanent_save:
        save_folder_name = seed.__str__()
    else:
        save_folder_name = "temp"

    save_folder = "../scenarios/scenario1/" + save_folder_name + "/"

    # save the ground truth and the measurements for the radar and the AIS
    store_object.store_object(truth, save_folder, "ground_truth.pk1")
    store_object.store_object(measurements_radar, save_folder, "measurements_radar.pk1")
    store_object.store_object(measurements_ais, save_folder, "measurements_ais.pk1")
    store_object.store_object(start_time, save_folder, "start_time.pk1")
    store_object.store_object(measurement_model_radar, save_folder, "measurement_model_radar.pk1")
    store_object.store_object(measurement_model_ais, save_folder, "measurement_model_ais.pk1")
    store_object.store_object(transition_model, save_folder, "transition_model.pk1")


