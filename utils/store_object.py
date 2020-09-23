""" save_objects: saves the object to a specified file using pickle

The functions receives objects which it stores to a given files using the pickle module.

"""
import pickle
import os


def store_object(obj, folder, filename):
    """

    :param obj:
    :param folder:
    :param filename:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
