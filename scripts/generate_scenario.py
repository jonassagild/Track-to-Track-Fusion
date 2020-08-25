"""Generates a simple one target scenario

Using stonesoup's functions, a simple one target scenario is generated. The target is randomly ...
TODO
"""
import numpy as np
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

start_time = datetime.now()

# specify seed to repeat example
np.random.seed(1996)

# combine two 1-D CV models to create a 2-D CV model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])

# starting at 0,0 and moving NE
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
# generate truth using transition_model and noise
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# plot the result
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.set_ylabel("$x$")
ax.set_xlabel("$y$")
ax.axis('equal')
ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")


def function():
    """

    :return:
    """
    return 0