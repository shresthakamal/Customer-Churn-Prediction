import logging

from churnprediction.config import config


class Log:
    @staticmethod
    def init():
        logging.basicConfig(
            level=logging.INFO,
            filename=config.LOG_FILE,
            format="%(levelname)s : %(asctime)s : %(message)s",
            datefmt="%d-%b-%Y %H:%M:%S",
        )
        return True
