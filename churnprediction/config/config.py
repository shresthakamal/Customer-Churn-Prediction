import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAME = "churn.csv"


DATA_PATH = os.path.join(BASE_DIR, "data")

DATASET_URL = "http://bit.ly/texam-2021"

DATASET_PATH = os.path.join(DATA_PATH, "raw", DATASET_NAME)


PRE_PROCESSED_DATA = os.path.join(DATA_PATH, "processed", "pre_processed_data.pkl")

CHECKPOINTS_PATH = os.path.join(BASE_DIR, "checkpoints")

LOG_FILE = os.path.join(CHECKPOINTS_PATH, "app.log")

TEST_SIZE = 0.3

RANDOM_STATE = 7

ESTIMATORS = 100

ALPHA = 0.7

FIGURE_PATH = os.path.join(BASE_DIR, "churnprediction/visualisation/")

# To Specify whether or not to train the model everytime
TRAIN_FLAG = True

LABELS = {"yes": 1, "no": 0}

FEATURES_TO_REMOVE = [
    "International_Minutes",
    "Evening_Minutes",
    "Night_Minutes",
    "Day_Minutes",
    "Voicemail_Plan",
]

COLUMNS = [
    "Account Length",
    "Voicemail Message",
    "Customer Service Calls",
    "International Plan",
    "Day Calls",
    "Day Charge",
    "Evening Calls",
    "Evening Charge",
    "Night Calls",
    "Night Charge",
    "International Calls",
    "International Charge",
    "Area Code",
]


if __name__ == "__main__":
    pass
