import logging
import os
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from churnprediction.config import config
from churnprediction.features.build_features import build_features
from churnprediction.utils.log import Log

# from churnprediction.utils.model_accuracy import model_accuracy


class TestModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.clf = TestModel.load_model(model_name)

    @staticmethod
    def load_model(model_name):
        file_path = os.path.join(
            config.CHECKPOINTS_PATH, "models/{}.pkl".format(model_name)
        )
        if os.path.exists(file_path):
            Log.init()
            logging.info("Testing with model: {}".format(model_name))

            with open(file_path, "rb") as handle:
                clf = pickle.load(handle)
            return clf
        else:
            print("[ERROR]: No saved model to predict, Turn on TRAIN_FLAG in config!!")
            return None

    def predict(self, **kwargs):
        if self.clf:
            prediction = self.clf.predict(kwargs["test_x"])
            return prediction
        else:
            return None


if __name__ == "__main__":

    tester = TestModel("GNB")

    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )

    predictions = tester.predict(test_x=x_test)

    print(f1_score(predictions, y_test))
