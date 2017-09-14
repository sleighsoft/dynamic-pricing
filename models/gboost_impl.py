from sklearn.ensemble import GradientBoostingClassifier


def get_model(conf):
    model = GradientBoostingClassifier(**conf)
    return model
