import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from tools import load_data


def find_model():
    X_train = load_data("X_train_f.csv")
    y_train = load_data("y_train_f.csv")
    X_test = load_data("X_test_f.csv")
    y_test = load_data("y_test_f.csv")

    dummy = DummyClassifier().fit(X_train, y_train)
    base_score = accuracy_score(y_train, dummy.predict(X_train))
    print(f"\tBase f1 score for seen data: {base_score:2.6f}")
