"""latex_utils module that contains latex functions

"""


def populate_latex_table(info_dict):
    """
    Populates a latex table similar to that of the results after performing Monte-Carlo sims on the fusion algorithms.
    """
    # find the noises
    process_noises = list(set([key[0] for key in info_dict.keys()]))
    radar_noises = list(set([key[1] for key in info_dict.keys()]))
    process_noises.sort()
    radar_noises.sort()

    latex_table = "& $\sigma_{radar}=" + str(radar_noises[0]) + "$ & $\sigma_{radar}=" + str(radar_noises[1]) + \
                  "$ & $\sigma_{radar}=" + str(radar_noises[2]) + "$ \\\\ \hline" + "\n"
    for process_noise in process_noises:
        latex_table += "$\sigma_{process}=" + str(process_noise) + "$"
        for radar_noise in radar_noises:
            latex_table += "& " + str(info_dict[(process_noise, radar_noise)])
        latex_table += "\\\\ \hline" + "\n"

