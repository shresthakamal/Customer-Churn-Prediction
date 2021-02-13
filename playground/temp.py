import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.metrics import f1_score

test_dataset = sys.argv[1]


def phone_split(phoneno):
    return int(phoneno.split("-")[0])


def test(test_dataset):

    df = pd.read_csv(test_dataset)

    drop_cols = [
        "Voicemail_Plan",
        "Day_Charge",
        "Evening_Charge",
        "Night_Charge",
        "International_Charge",
    ]
    df.drop(drop_cols, inplace=True, axis=1)

    le = preprocessing.LabelEncoder()

    df["phone_code"] = np.zeros((len(df), 1))
    for i in range(len(df)):
        df["phone_code"][i] = phone_split(df["Phone"][i])

    df.drop("Phone", inplace=True, axis=1)

    df["Churn"].replace("yes", 1, inplace=True)
    df["Churn"].replace("no", 0, inplace=True)

    y_test = df["Churn"]

    df.drop("Churn", inplace=True, axis=1)

    # Label Encoding as in Training
    # Internation_Plan
    internation_plan = open("./Models/internation_plan.pkl", "rb")
    le_internation_plan = pickle.load(internation_plan)
    df["International_Plan"] = le_internation_plan.fit_transform(
        df["International_Plan"]
    )

    # State
    state = open("./Models/state.pkl", "rb")
    le_state = pickle.load(state)
    state.close()
    df["State"] = le_state.transform(df["State"])

    with open("./Models/rfModel.pkl", "rb") as f:
        rfc = pickle.load(f)

    preds = rfc.predict(df)

    f1score_RFC = f1_score(y_test, preds)

    print("************************************************")
    print(f"Random Forest F1 Score: {f1score_RFC}")
    print("************************************************")

    print("************************************************")

    def result(p):
        if p == 1:
            return "yes"
        else:
            return "no"

    res = [result(out) for out in preds]
    print(f"Predictions from Random Forest: {res}")
    print("************************************************")


test(sys.argv[1])
