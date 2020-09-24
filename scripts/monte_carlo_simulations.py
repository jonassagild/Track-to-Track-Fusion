"""monte_carlo_simulations Script for running monte carlo simulations

Script used for running monte carlo simulations.
"""


from utils import open_object
from utils import calc_metrics

from scripts.generate_scenario import generate_scenario

num_mc_iterations = 4

# generate scenario
seed = 2000
generate_scenario(seed=seed, permanent_save=False)

# load ground truth and the measurements
data_folder = "../scenarios/scenario1/" + str(seed) + "/"
ground_truth = open_object.open_object(data_folder + "ground_truth.pk1")
measurements_radar = open_object.open_object(data_folder + "measurements_radar.pk1")
measurements_ais = open_object.open_object(data_folder + "measurements_ais.pk1")

# load start_time
start_time = open_object.open_object(data_folder + "start_time.pk1")

