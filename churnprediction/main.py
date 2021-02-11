import os

from MultiChoice import MultiChoice
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from churnprediction.config import config
from churnprediction.features.build_features import build_features
from churnprediction.models import test_model, train_model
from churnprediction.utils.generate_roc import generate_roc


def train_pipeline(model_name, x_train, y_train, x_test, y_test):

    trainer = train_model.TrainModel(model_name)

    clf = trainer.fit(train_x=x_train, train_y=y_train)

    predictions = clf.predict(x_test)

    f1 = f1_score(predictions, y_test)

    if generate_roc(y_test, predictions, model_name):
        print("[INFO]: ROC Curve Generated!")

    return f1


def test_pipeline(model_name, test_query):

    tester = test_model.TestModel(model_name)

    prediction = tester.predict(test_x=test_query)

    for key, value in config.LABELS.items():
        if value == prediction[0]:
            churn = key
            print("[RESULTS: (main/test_pipeline): Predicted Churn: {}]".format(key))

    return churn


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = build_features(
        dataset_path=os.path.join(config.DATA_PATH, "raw", config.DATASET_NAME),
        split_ratio=config.TEST_SIZE,
    )
    print("Available Models:" "K-nearest Neighbour")
    # User's Model Choice
    user_selected_model = MultiChoice(
        "Select one of the following models:",
        options=("knn", "lr", "svm", "rf", "dt", "bagging", "gboost", "xgboost", "GNB"),
    )().lower()

    print(
        "[INFO]: F1 Score: {}".format(
            train_pipeline(user_selected_model, x_train, y_train, x_test, y_test)
        )
    )

    columns = [
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

    user_data = []

    for name in columns:
        var = float(input("Enter {}: ".format(name)))
        user_data.append(var)

    test_pipeline(
        user_selected_model,
        [user_data],
    )
