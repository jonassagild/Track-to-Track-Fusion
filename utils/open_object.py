""" open_objects: opens the object in a specified file using pickle

The functions receives filename and opens the object store in the file using the pickle module.

"""
import pickle


def open_object(filepath):
    with open(filepath, 'rb') as inp:
        obj = pickle.load(inp)
        return obj
