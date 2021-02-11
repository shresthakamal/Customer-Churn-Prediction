import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from churnprediction.config import config
from churnprediction.utils.seralizer import save_object


def build_features(dataset_path, split_ratio=0.3):

    data = pd.read_csv(dataset_path)

    df = data.drop(["State", "Phone"], axis=1)

    labels = config.LABELS

    categorical_columns = [
        col for col in df.columns.tolist() if df[col].dtype == "object"
    ]

    for col in categorical_columns:
        df[col] = df[col].map(labels)

    features_to_remove = config.FEATURES_TO_REMOVE

    df = df.drop(features_to_remove, axis=1)

    """DATA PRE-PROCESSING

    No missing data
    No null values
    Encoding of categorical values needed
    No Standaridastaion needed
    No ill formated values

    """

    train, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["Churn"])

    train_x = train.loc[:, train.columns != "Churn"]
    test_x = test.loc[:, test.columns != "Churn"]

    train_y = train.loc[:, train.columns == "Churn"]
    test_y = test.loc[:, test.columns == "Churn"]

    # Upsampling using SMOTE
    sm = SMOTE()
    train_x, train_y = sm.fit_resample(train_x, train_y)
    test_x, test_y = sm.fit_resample(test_x, test_y)

    save_object(
        filepath=os.path.join(config.DATA_PATH, "processed"),
        filename="pre_processed_data",
        object_arr=[df],
    )

    save_object(
        filepath=os.path.join(config.CHECKPOINTS_PATH),
        filename="train_data",
        object_arr=[train],
    )

    save_object(
        filepath=os.path.join(config.CHECKPOINTS_PATH),
        filename="test_data",
        object_arr=[test],
    )

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )
