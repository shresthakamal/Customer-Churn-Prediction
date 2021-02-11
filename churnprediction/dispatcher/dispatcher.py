from sklearn import svm
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churnprediction.config import config

# The Parameters can be more manged // NO TIME TO DO THIS

MODELS = {
    "lr": LogisticRegression(max_iter=50, penalty="l2"),
    "svm": svm.SVC(),
    "dt": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
    "knn": KNeighborsClassifier(),
    "bagging": BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        n_estimators=config.ESTIMATORS,
        random_state=config.RANDOM_STATE,
    ),
    "rf": RandomForestClassifier(
        n_estimators=config.ESTIMATORS, random_state=config.RANDOM_STATE
    ),
    "xgboost": XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False),
    "gboost": GradientBoostingClassifier(
        n_estimators=config.ESTIMATORS,
        learning_rate=config.ALPHA,
        max_depth=1,
        random_state=config.RANDOM_STATE,
    ),
    "gnb": GaussianNB(),
}


if __name__ == "__main__":
    pass
