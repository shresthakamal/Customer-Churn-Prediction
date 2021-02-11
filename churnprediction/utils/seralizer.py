import logging
import os
import pickle

from churnprediction.utils.log import Log


def save_object(filepath, filename, object_arr):

    Log.init()

    filepath = os.path.join(filepath, f"{filename}.pkl")

    logging.info("Saving {}".format(filename))

    with open(filepath, "wb") as handle:
        pickle.dump(object_arr[0], handle)
