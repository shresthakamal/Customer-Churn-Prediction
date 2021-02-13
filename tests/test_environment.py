import os
import logging
import pickle
from churnprediction.features import build_features
from churnprediction.models.test_model import TestModel
from tests import config
from churnprediction.utils.log import Log
from churnprediction.utils.seralizer import save_object
from churnprediction.utils.get_user_data import get_user_data
from churnprediction.utils import get_user_selected_model
from churnprediction.main import train_pipeline, test_pipeline


def build(dataset_path):

    logging.info(
        "[INFO]: Pre-Processing Testing Dataset using:  {}".format(dataset_path)
    )

    df = build_features.pre_precessing(dataset_path)
    print("[INFO]: Data Pre-Processing Completed !!")


def load_processed_dataframe(filepath=config.PRE_PROCESSED_DATA):
    with open(filepath, "rb") as handle:
        df = pickle.load(handle)
    return df


def test(model=config.DEFAULT_TEST_MODEL):

    df = load_processed_dataframe()

    logging.info("[INFO]: Testing on user selected model:  {}".format(model))

    test_x = df.loc[:, df.columns != "Churn"]

    tester = TestModel(model)

    predictions = tester.predict(test_x=test_x)

    logging.info("[INFO]: Displaying Predictions using: {}  model".format(model))

    return predictions


if __name__ == "__main__":
    Log.init()
    dataset_path = os.path.join(config.TEST_DATASET_PATH)

    build(dataset_path)

    user_selected_model = get_user_selected_model.get_user_selected_model()

    print("[INFO]: Predictions on the Test Dataset: ")

    print(test(user_selected_model))
