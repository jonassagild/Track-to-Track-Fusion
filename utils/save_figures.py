"""save_figures

provides function to save figure, and create folder if it does not exists. Uses matplotlib savefig to save.

"""

import os
import matplotlib.pyplot as plt


def save_figure(folder_path, name, fig):
    """
    Saves the figure using matplotlibs savefig, and creates the folder in path if it does not exists.
    :param folder_path: the path to the folder in which to save the figure. Assumes no trailing '/'
    :param name:
    :param fig:
    :return: nothing
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    fig.savefig(folder_path + "/" + name)
