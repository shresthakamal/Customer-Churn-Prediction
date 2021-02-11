import logging
import os

import requests
from tqdm import tqdm

from churnprediction.config import config
from churnprediction.utils import log


def download_dataset(url):

    log.Log.init()

    dataset_filepath = os.path.join(
        config.BASE_DIR, config.DATA_PATH, "raw", config.DATASET_NAME
    )

    try:
        response = requests.get(url, stream=True)
        logging.info("Dataset Successfully Downloaded !!")

        with open(dataset_filepath, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    except requests.exceptions.RequestException as e:
        logging.error("Dataset Download Failed, Exception:{}".format(e))

    return True


if __name__ == "__main__":
    print(download_dataset(url=config.DATASET_URL))
