"""monte_carlo_simulations Script for running monte carlo simulations

Script used for running monte carlo simulations.
"""
import numpy as np
import scipy
from stonesoup.types.state import GaussianState
from matplotlib import pyplot as plt

from utils import open_object
from utils import calc_metrics

from scripts.generate_scenario import generate_scenario

from trackers.kalman_filter_view_AIS_as_measurement import kalman_filter_ais_as_measurement

# loop starts here
# loop over seeds?
start_seed = 50
end_seed = 300
num_mc_iterations = end_seed - start_seed

stats_overall = []
for seed in range(start_seed, end_seed):
    # generate scenario
    generate_scenario(seed=seed, permanent_save=False, sigma_transition=0.01, sigma_meas_radar=3, sigma_meas_ais=1)

    folder = "temp"  # temp instead of seed, as it is not a permanent save

    # load ground truth and the measurements
    data_folder = "../scenarios/scenario1/" + folder + "/"
    ground_truth = open_object.open_object(data_folder + "ground_truth.pk1")
    measurements_radar = open_object.open_object(data_folder + "measurements_radar.pk1")
    measurements_ais = open_object.open_object(data_folder + "measurements_ais.pk1")

    # load start_time
    start_time = open_object.open_object(data_folder + "start_time.pk1")

    # prior
    prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5])**2, timestamp=start_time)

    kf_ais_as_measurement = kalman_filter_ais_as_measurement(measurements_radar, measurements_ais, start_time, prior,
                                             sigma_process=0.01,
                               sigma_meas_radar=2.5, sigma_meas_ais=0.5)

    tracks_fused, tracks_radar = kf_ais_as_measurement.track()

    # Calculate some metrics
    stats_individual = {}
    # calculate NEES
    stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused, ground_truth)
    # calculate ANEES
    stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])

    stats_individual["seed"] = seed

    stats_overall.append(stats_individual)

# plot some results
num_tracks = len(tracks_fused)
alpha = 0.95
ci_nees = scipy.stats.chi2.interval(alpha, 4)
ci_anees = np.array(scipy.stats.chi2.interval(alpha, 4*num_tracks)) / num_tracks

# plot ANEES and confidence interval
fig_ci_anees = plt.figure(figsize=(10, 6))
ax_ci_anees = fig_ci_anees.add_subplot(1, 1, 1)
ax_ci_anees.set_xlabel("$x$")
ax_ci_anees.set_ylabel("$y$")

# plot upper and lower confidence intervals
ax_ci_anees.plot([0, num_mc_iterations], [ci_anees[0], ci_anees[0]], color='red')
ax_ci_anees.plot([0, num_mc_iterations], [ci_anees[1], ci_anees[1]], color='red')

anees = [stat['ANEES'] for stat in stats_overall]

# plot the ANEES values
ax_ci_anees.plot(list(range(0, num_mc_iterations)), anees, marker='+', ls='None', color='blue')

fig_ci_anees.show()
