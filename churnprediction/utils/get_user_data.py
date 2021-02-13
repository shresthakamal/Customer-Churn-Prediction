from churnprediction.dispatcher import dispatcher
from MultiChoice import MultiChoice
from churnprediction.config import config


def get_user_data():
    # User's Model Choice
    user_selected_model = MultiChoice(
        "Select one of the following models:",
        options=(dispatcher.MODELS.keys()),
    )().lower()

    columns = config.COLUMNS

    user_data = []

    for name in columns:
        var = float(input("Enter {}: ".format(name)))
        user_data.append(var)

    return user_selected_model, user_data


if __name__ == "__main__":
    get_user_data()
