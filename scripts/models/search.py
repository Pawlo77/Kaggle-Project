import sys
import os
import optuna
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
optuna.logging.set_verbosity(optuna.logging.ERROR)

from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from .definitions import (
    set_objective,
    _LOGISTIC,
    _SVC,
    _MLP,
    _XGB,
    NAMES,
    SCORES,
    RANDOM_STATE,
)
from tools import load_data, save_object


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


def get_model(classifier_name):
    assert classifier_name in NAMES, "Classifier not recognized"

    if classifier_name == _LOGISTIC:
        return LogisticRegression
    if classifier_name == _SVC:
        return SVC
    if classifier_name == _MLP:
        return MLPClassifier
    # _XGB:
    return xgb.XGBClassifier


def find_model(study_name="banking"):
    X_train = load_data("X_train_f.csv").to_numpy()
    y_train = load_data("y_train_f.csv").to_numpy().ravel()
    X_test = load_data("X_test_f.csv").to_numpy()
    y_test = load_data("y_test_f.csv").to_numpy().ravel()
    d_matrix = xgb.DMatrix(X_train, label=y_train)

    # list of pais model object, model training score
    models = []

    for classifier_name, n_trials in [
        (_XGB, 500),
        (_MLP, 30),
        (_SVC, 50),
        (_LOGISTIC, 500),
    ]:
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///db.sqlite3",
            study_name=study_name + f"_{classifier_name}",
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            load_if_exists=True,
        )
        objective = partial(
            set_objective,
            X_train=X_train,
            y_train=y_train,
            classifier_name=classifier_name,
            d_matrix=d_matrix,
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, gc_after_trial=False)

        model = get_model(classifier_name)
        model = model(**study.best_params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        scores_train = get_scores(y_train, y_pred_train)
        scores_test = get_scores(y_test, y_pred_test)
        print_scores(model.__class__, scores_train, scores_test)
        models.append((classifier_name, model, scores_test[SCORES[0].__name__]))

    models.sort(key=lambda x: x[-1], reverse=True)
    print(f"Models order (based on {SCORES[0].__name__}):")
    for model in models:
        print(f"\t- {model[0]}")
        save_object(model[1], f"model_{model[0]}.pkl")
