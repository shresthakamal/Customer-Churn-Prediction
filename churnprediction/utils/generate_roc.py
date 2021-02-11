import os

import matplotlib.pyplot as plt
from sklearn import metrics

from churnprediction.config import config


def generate_roc(test_y, prediction, model_name):
    fpr, tpr, _ = metrics.roc_curve(test_y, prediction)
    auc = metrics.roc_auc_score(test_y, prediction)
    plt.plot(fpr, tpr, label="{}".format(model_name))
    plt.legend(loc=4)
    plt.savefig(os.path.join(config.FIGURE_PATH) + "{}.png".format(model_name))


if __name__ == "__main__":
    pass
