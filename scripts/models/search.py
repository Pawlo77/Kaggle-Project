import sys
import os
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, roc_auc_score

from tools import load_data

RANDOM_STATE = 42
LOGISTIC_PARAMS = {"max_iter": 1000, "random_state": RANDOM_STATE, "n_jobs": -1}
TREE_PARAMS = {
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
SVC_PARAMS = {"random_state": RANDOM_STATE}
MLP_PARAMS = {
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
    "early_stopping": True,
    "learning_rate": "adaptive",
}
SCORES = [f1_score, roc_auc_score]


def get_scores(y_true, y_pred):
    scores = {}
    for score in SCORES:
        scores[score.__name__] = score(y_true, y_pred)
    return scores


def print_scores(name, scores_train, scores_test):
    print(f"\tEvaluating {str(name).split('.')[-1][:-2]}:")
    for name, score in (("traing", scores_train), ("test", scores_test)):
        print(f"\t- Scores on {name} set:")
        for score_name, score_val in score.items():
            print(f"\t\t- {score_name} - {score_val:2.4f}")
    print()


def find_model():
    X_train = load_data("X_train_f.csv").to_numpy()
    y_train = load_data("y_train_f.csv").to_numpy().ravel()
    X_test = load_data("X_test_f.csv").to_numpy()
    y_test = load_data("y_test_f.csv").to_numpy().ravel()

    # list of pais model object, model training score
    models = []

    for model, params in (
        (LogisticRegression, LOGISTIC_PARAMS),
        (xgb.XGBClassifier, TREE_PARAMS),
        (SVC, SVC_PARAMS),
        (MLPClassifier, MLP_PARAMS),
    ):
        model = model(**params)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        scores_train = get_scores(y_train, y_pred_train)
        scores_test = get_scores(y_test, y_pred_test)
        print_scores(model.__class__, scores_train, scores_test)
        models.append((model, scores_test))
