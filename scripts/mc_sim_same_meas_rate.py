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
from utils.latex_utils import populate_latex_table

# seeds
start_seed = 0
end_seed = 100  # normally 500
num_mc_iterations = end_seed - start_seed

# params
save_fig = False
print_latex = False

# scenario parameters
sigma_process_list = [3]  # [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 3, 3, 3]  # [0.05, 0.05]  #
sigma_meas_radar_list = [200]  # [1, 5, 20, 1, 5, 20, 1, 5, 20]  # [5, 20]  #
sigma_meas_ais_list = sigma_meas_radar_list
num_steps = 200
num_steps_metrics = 150

# dicts to store final results for printing in a latex friendly way
aanees_overall = {}
rmse_overall = {}

for sigma_process, sigma_meas_radar, sigma_meas_ais in zip(sigma_process_list, sigma_meas_radar_list,
                                                           sigma_meas_ais_list):
    stats_overall = []
    for seed in range(start_seed, end_seed):
        # generate scenario
        generate_scenario_2(seed=seed, permanent_save=False, sigma_process=sigma_process,
                            sigma_meas_radar=sigma_meas_radar, sigma_meas_ais=sigma_meas_ais, timesteps=num_steps)

        folder = "temp"  # temp instead of seed, as it is not a permanent save

        # load ground truth and the measurements
        data_folder = "../scenarios/scenario2/" + folder + "/"
        ground_truth = open_object.open_object(data_folder + "ground_truth.pk1")
        measurements_radar = open_object.open_object(data_folder + "measurements_radar.pk1")
        measurements_ais = open_object.open_object(data_folder + "measurements_ais.pk1")

        # load start_time
        start_time = open_object.open_object(data_folder + "start_time.pk1")

        # prior
        initial_covar = np.diag([sigma_meas_radar * sigma_meas_ais, sigma_meas_radar*sigma_process,
                                 sigma_meas_radar * sigma_meas_ais, sigma_meas_radar*sigma_process]) ** 2
        prior = GaussianState([1, 1.1, -1, 0.9], initial_covar, timestamp=start_time)

        # trackers
        kf_ais_as_measurement = kalman_filter_ais_as_measurement(measurements_radar, measurements_ais, start_time,
                                                                 prior,
                                                                 sigma_process=sigma_process,
                                                                 sigma_meas_radar=sigma_meas_radar,
                                                                 sigma_meas_ais=sigma_meas_ais)
        kf_independent_fusion = kalman_filter_independent_fusion(start_time,
                                                                 prior,
                                                                 sigma_process_radar=sigma_process,
                                                                 sigma_process_ais=sigma_process,
                                                                 sigma_meas_radar=sigma_meas_radar,
                                                                 sigma_meas_ais=sigma_meas_ais)
        kf_dependent_fusion = kalman_filter_dependent_fusion(measurements_radar, measurements_ais, start_time, prior,
                                                             sigma_process_radar=sigma_process,
                                                             sigma_process_ais=sigma_process,
                                                             sigma_meas_radar=sigma_meas_radar,
                                                             sigma_meas_ais=sigma_meas_ais)

        tracks_fused_independent, _, _ = kf_independent_fusion.track(measurements_radar, measurements_ais)
        tracks_fused_dependent, _, _ = kf_dependent_fusion.track()
        tracks_fused_ais_as_measurement, _ = kf_ais_as_measurement.track()

        # fix the length of the fusions to dependent (as it is a bit shorter)
        num_tracks = len(tracks_fused_dependent)
        # get the num_tracks last elements
        tracks_fused_independent = tracks_fused_independent[-num_tracks:]
        tracks_fused_ais_as_measurement = tracks_fused_ais_as_measurement[-num_tracks:]
        # remove the first element of ground_truth (because we only fuse the n-1 last with dependent fusion)
        ground_truth = ground_truth[-num_tracks:]

        # Calculate some metrics
        stats_individual = {}
        # calculate NEES
        stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_independent[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        # calculate ANEES
        stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # calculate RMSE
        stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_fused_independent[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        stats_individual["seed"] = seed
        stats_individual["fusion_type"] = "independent"
        stats_overall.append(stats_individual)

        stats_individual = {}
        # calculate NEES
        stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_dependent[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        # calculate ANEES
        stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # calculate RMSE
        stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_fused_dependent[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        stats_individual["seed"] = seed
        stats_individual["fusion_type"] = "dependent"
        stats_overall.append(stats_individual)

        stats_individual = {}
        # calculate NEES
        stats_individual["NEES"] = calc_metrics.calc_nees(tracks_fused_ais_as_measurement[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        # calculate ANEES
        stats_individual["ANEES"] = calc_metrics.calc_anees(stats_individual["NEES"])
        # calculate RMSE
        stats_individual["RMSE"] = calc_metrics.calc_rmse(tracks_fused_ais_as_measurement[-num_steps_metrics:],
                                                          ground_truth[-num_steps_metrics:])
        stats_individual["seed"] = seed
        stats_individual["fusion_type"] = "ais as measurement"
        stats_overall.append(stats_individual)

    # plot some results
    num_tracks = num_steps_metrics
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

    rmse_independent = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "independent"])
    rmse_dependent = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "dependent"])
    rmse_ais_as_measurement = np.mean([stat['RMSE'] for stat in stats_overall if stat["fusion_type"] == "ais as " 
                                                                                                        "measurement"])

    # plot the ANEES values
    ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_independent, marker='+', ls='None', color='blue',
                     label='Independent')
    ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_dependent, marker='+', ls='None', color='red',
                     label='Dependent')
    ax_ci_anees.plot(list(range(start_seed, end_seed)), anees_ais_as_measurement, marker='+', ls='None', color='green',
                     label='AIS as measurement')

    ax_ci_anees.legend()
    # fig_ci_anees.show()

    # save_figure("../results/scenario2/1996", "monte_carlo_all_trackers_same_params.svg", fig_ci_anees)

    # plot to show how the average ANEES approaches some limit. Plot a changing confidence interval?
    fig_ci_average_anees = plt.figure(figsize=(10, 6))
    ax_ci_average_anees = fig_ci_average_anees.add_subplot(1, 1, 1)
    ax_ci_average_anees.set_xlabel("MC iterations")
    ax_ci_average_anees.set_ylabel("ANEES")
    ax_ci_average_anees.set_ylim(2.5, 6.5)

    ci_average_anees = [
        np.array(scipy.stats.chi2.interval(alpha, 4 * num_tracks * num_MC_it)) / (num_tracks * num_MC_it)
        for num_MC_it in range(1, end_seed - start_seed + 1)]
    ci_average_anees_lower = [ci[0] for ci in ci_average_anees]
    ci_average_anees_upper = [ci[1] for ci in ci_average_anees]

    # calculate average anees
    # not the quickest, but works
    average_anees_independent_list = [np.average(anees_independent[:idx]) for idx in range(1, num_mc_iterations + 1)]
    average_anees_dependent_list = [np.average(anees_dependent[:idx]) for idx in range(1, num_mc_iterations + 1)]
    average_anees_ais_as_measurement_list = [np.average(anees_ais_as_measurement[:idx])
                                             for idx in range(1, num_mc_iterations + 1)]

    # store anees and rmse in dicts
    aanees_overall[(sigma_process, sigma_meas_radar, "indep")] = average_anees_independent_list[-1]
    aanees_overall[(sigma_process, sigma_meas_radar, "dep")] = average_anees_dependent_list[-1]
    aanees_overall[(sigma_process, sigma_meas_radar, "kf")] = average_anees_ais_as_measurement_list[-1]
    rmse_overall[(sigma_process, sigma_meas_radar, "indep")] = rmse_independent
    rmse_overall[(sigma_process, sigma_meas_radar, "dep")] = rmse_dependent
    rmse_overall[(sigma_process, sigma_meas_radar, "kf")] = rmse_ais_as_measurement

    # plot upper and lower confidence intervals
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), ci_average_anees_lower, color='black',
                             label='Confidence Intervals')
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), ci_average_anees_upper, color='black')

    # plot the average ANEES
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_independent_list, color='blue',
                             label='Independent')
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_dependent_list, color='red',
                             label='Dependent')
    ax_ci_average_anees.plot(list(range(start_seed, end_seed)), average_anees_ais_as_measurement_list, color='green',
                             label='KF fusion')

    # set title
    title = "ANEES with parameters $\sigma_{process}=$" + str(sigma_process) + ",  $\sigma_{AIS}=$" + \
            str(sigma_meas_ais) + ", $\sigma_{Radar}=$" + str(sigma_meas_radar)
    ax_ci_average_anees.set_title(title, fontsize=20)

    ax_ci_average_anees.legend(prop={'size': 12})
    fig_ci_average_anees.show()

    if save_fig:
        fig_name = "process_" + str(sigma_process) + "_AIS_" + str(sigma_meas_ais) + "_Radar_" \
                   + str(sigma_meas_radar) + ".pdf"
        save_figure("../results/final_results/scenario1/mc_average_anees", fig_name, fig_ci_average_anees)

    # print some results
    print("Process: " + str(sigma_process) + " AIS: " + str(sigma_meas_ais) + " Radar: " + str(sigma_meas_radar))
    print("Average ANEES independent: " + str(average_anees_independent_list[-1]))
    print("Average ANEES dependent: " + str(average_anees_dependent_list[-1]))
    print("Average ANEES AIS as measurement: " + str(average_anees_ais_as_measurement_list[-1]))
    print("RMSE independent: " + str(rmse_independent))
    print("RMSE dependent: " + str(rmse_dependent))
    print("RMSE AIS as measurement: " + str(rmse_ais_as_measurement))
    print("")

if print_latex:
    fusion_types = ["indep", "dep", "kf"]
    for fusion_type in fusion_types:
        print("AANEES")
        print(fusion_type + "\n" + populate_latex_table(aanees_overall, fusion_type))
        print("")
        print("RMSE")
        print(fusion_type + "\n" + populate_latex_table(rmse_overall, fusion_type))
        print("")

print("CI intervals: " + str(ci_average_anees_lower[-1]) + ", " + str(ci_average_anees_upper[-1]))
