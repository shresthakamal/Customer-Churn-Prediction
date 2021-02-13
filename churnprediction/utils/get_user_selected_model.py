from churnprediction.dispatcher import dispatcher
from MultiChoice import MultiChoice


def get_user_selected_model():
    user_selected_model = MultiChoice(
        "Select one of the following models:",
        options=(dispatcher.MODELS.keys()),
    )().lower()
    print("[INFO]: Selected Model: {}".format(user_selected_model))
    return user_selected_model


if __name__ == "__main__":
    get_user_selected_model()
