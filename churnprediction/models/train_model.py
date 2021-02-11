import logging
import os
import pickle

from churnprediction.config import config
from churnprediction.dispatcher import dispatcher
from churnprediction.features.build_features import build_features
from churnprediction.utils.log import Log
from churnprediction.utils.seralizer import save_object


class TrainModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = dispatcher.MODELS[model_name]

    def fit(self, **kwargs):

        Log.init()
        logging.info("Training with model: {}".format(self.model_name))

        clf = self.model.fit(kwargs["train_x"], kwargs["train_y"].values.ravel())

        save_object(
            filepath=os.path.join(config.CHECKPOINTS_PATH, "models"),
            filename="{}".format(self.model_name),
            object_arr=[clf],
        )
        return clf


if __name__ == "__main__":

    trainer = TrainModel("GNB")

    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )

    print(trainer.fit(train_x=x_train, train_y=y_train))
