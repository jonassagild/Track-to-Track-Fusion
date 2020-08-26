""" save_objects: saves the object to a specified file using pickle

The functions receives objects which it stores to a given files using the pickle module.

"""
import pickle


def store_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
