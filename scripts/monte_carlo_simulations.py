"""monte_carlo_simulations Script for running monte carlo simulations

Script used for running monte carlo simulations.
"""
import numpy as np
import scipy
from stonesoup.types.state import GaussianState
from matplotlib import pyplot as plt

from utils import open_object
from utils import calc_metrics

from utils.scenario_generator import generate_scenario_2
from trackers.kalman_filter_view_AIS_as_measurement import kalman_filter_ais_as_measurement
from trackers.kalman_filter_independent_fusion import kalman_filter_independent_fusion
from trackers.kalman_filter_dependent_fusion import kalman_filter_dependent_fusion

from utils.save_figures import save_figure

# loop starts here
# loop over seeds?
start_seed = 50
end_seed = 300
num_mc_iterations = end_seed - start_seed

stats_overall = []
for seed in range(start_seed, end_seed):
    # generate scenario
    generate_scenario_2(seed=seed, permanent_save=False, sigma_transition=0.01, sigma_meas_radar=3, sigma_meas_ais=1)

    folder = "temp"  # temp instead of seed, as it is not a permanent save

    # load ground truth and the measurements
    data_folder = "../scenarios/scenario2/" + folder + "/"
    ground_truth = open_object.open_object(data_folder + "ground_truth.pk1")
    measurements_radar = open_object.open_object(data_folder + "measurements_radar.pk1")
    measurements_ais = open_object.open_object(data_folder + "measurements_ais.pk1")

    # remove the first element of ground_truth (because we only fuse the n-1 last with dependent fusion)
    ground_truth = ground_truth[-(len(ground_truth)-1):]

    # load start_time
    start_time = open_object.open_object(data_folder + "start_time.pk1")

    # prior
    prior = GaussianState([0, 1, 0, 1], np.diag([1.5, 0.5, 1.5, 0.5]) ** 2, timestamp=start_time)

    # trackers
    kf_ais_as_measurement = kalman_filter_ais_as_measurement(measurements_radar, measurements_ais, start_time, prior,
                                                             sigma_process=0.01, sigma_meas_radar=3.5,
                                                             sigma_meas_ais=1.3)
    kf_independent_fusion = kalman_filter_independent_fusion(measurements_radar, measurements_ais, start_time, prior,
                                                             sigma_process_radar=0.01, sigma_process_ais=0.01,
                                                             sigma_meas_radar=3.5, sigma_meas_ais=1.3)
    kf_dependent_fusion = kalman_filter_dependent_fusion(measurements_radar, measurements_ais, start_time, prior,
                                                         sigma_process_radar=0.01, sigma_process_ais=0.02,
                                                         sigma_meas_radar=3.5, sigma_meas_ais=1.3)

    tracks_fused_independent, _, _ = kf_independent_fusion.track()
    tracks_fused_dependent, _, _ = kf_dependent_fusion.track()
    tracks_fused_ais_as_measurement, _ = kf_ais_as_measurement.track()

    # fix the length of the fusions to dependent (as it is a bit shorter)
    num_tracks = len(tracks_fused_dependent)
    # get the num_tracks last elements
    tracks_fused_independent = tracks_fused_independent[-num_tracks:]
    tracks_fused_ais_as_measurement = tracks_fused_ais_as_measurement[-num_tracks:]

    # Calculate some metrics
    stats_individual = {}
    # calculate NEES
    stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_independent, ground_truth)
    # calculate ANEES
    stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
    stats_individual["seed"] = seed
    stats_individual["fusion_type"] = "independent"
    stats_overall.append(stats_individual)

    stats_individual = {}
    # calculate NEES
    stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_dependent, ground_truth)
    # calculate ANEES
    stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
    stats_individual["seed"] = seed
    stats_individual["fusion_type"] = "dependent"
    stats_overall.append(stats_individual)

    stats_individual = {}
    # calculate NEES
    stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_ais_as_measurement, ground_truth)
    # calculate ANEES
    stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
    stats_individual["seed"] = seed
    stats_individual["fusion_type"] = "ais as measurement"
    stats_overall.append(stats_individual)

# plot some results
num_tracks = len(tracks_fused_independent)
alpha = 0.95
ci_nees = scipy.stats.chi2.interval(alpha, 4)
ci_anees = np.array(scipy.stats.chi2.interval(alpha, 4 * num_tracks)) / num_tracks

# plot ANEES and confidence interval
fig_ci_anees = plt.figure(figsize=(10, 6))
ax_ci_anees = fig_ci_anees.add_subplot(1, 1, 1)
ax_ci_anees.set_xlabel("MC seed")
ax_ci_anees.set_ylabel("ANEES")

# plot upper and lower confidence intervals
ax_ci_anees.plot([start_seed, end_seed], [ci_anees[0], ci_anees[0]], color='red')
ax_ci_anees.plot([start_seed, end_seed], [ci_anees[1], ci_anees[1]], color='red')

anees_independent = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "independent"]
anees_dependent = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "dependent"]
anees_ais_as_measurement = [stat['ANEES'] for stat in stats_overall if stat["fusion_type"] == "ais as measurement"]

# plot the ANEES values
ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_independent, marker='+', ls='None', color='blue',
                 label='Independent')
ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_dependent, marker='+', ls='None', color='red',
                 label='Dependent')
ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_ais_as_measurement, marker='+', ls='None', color='green',
                 label='AIS as measurement')

ax_ci_anees.legend()
fig_ci_anees.show()

save_figure("../results/scenario2/1996", "monte_carlo.pdf", fig_ci_anees)
