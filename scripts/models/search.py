import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn.dummy import DummyClassifier

from tools import load_data


def find_model():
    X_train = load_data("X_train_imputed_special.csv")
    y_train = load_data("y_train.csv")

    return DummyClassifier()
