"""Generates a simple one target scenario

Using stonesoup's functions, a simple one target scenario is generated. The target is randomly ...
TODO

Inspiration for how to generate ground truth and measurements are taken from
https://stonesoup.readthedocs.io/en/latest/auto_tutorials/01_KalmanFilterTutorial.html#sphx-glr-auto-tutorials-01-kalmanfiltertutorial-py
"""

from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import uniform

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection, Detection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian

start_time = datetime.now()

# specify seed to be able repeat example
np.random.seed(1996)

# combine two 1-D CV models to create a 2-D CV model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01), ConstantVelocity(0.01)])

# starting at 0,0 and moving NE
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# generate truth using transition_model and noise
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))

# plot the result
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("$x$")
ax.set_xlabel("$y$")
ax.axis('equal')
ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")
# fig.show()

# print(transition_model.covar(time_interval=timedelta(seconds=1)))
# print(transition_model.matrix(time_interval=timedelta(seconds=1)))

# Simulate measurements
# Specify measurement model
measurement_model = LinearGaussian(
    ndim_state=4,  # number of state dimensions
    mapping=(0, 2),  # mapping measurement vector index to state index
    noise_covar=np.array([[3, 0],  # covariance matrix for Gaussian PDF
                          [0, 3]])
)

# generate measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# plot the result
ax.scatter([state.state_vector[0] for state in measurements],
           [state.state_vector[1] for state in measurements],
           color='b')
fig.show()

# TODO: save ground truth along with two sets of noisy measurements.

# TODO: save ground truth along with two sets of noisy measurements, where the sampling rate is different for the two
#  sets of data
