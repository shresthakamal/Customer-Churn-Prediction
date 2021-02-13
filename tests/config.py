import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data")

PRE_PROCESSED_DATA = os.path.join(DATA_PATH, "processed", "pre_processed_data.pkl")


# TESTS CONFIG

# Place holdout set on data/raw
# Rename it to "test.csv"


TEST_DATASET_NAME = "test.csv"

TEST_DATASET_PATH = os.path.join(DATA_PATH, "raw", TEST_DATASET_NAME)

DEFAULT_TEST_MODEL = "rf"
