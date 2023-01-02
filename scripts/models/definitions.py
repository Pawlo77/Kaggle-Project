import random
import optuna
import xgboost as xgb
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

RANDOM_STATE = 42

_LOGISTIC = "LogisticRegression"
_SVC = "SVC"
_MLP = "MLPClassifier"
_XGB = "XGBClassifier"

SCORES = [f1_score, roc_auc_score]

PARAMS = {}
PARAMS[_LOGISTIC] = {
    "max_iter": 100000,
    "random_state": RANDOM_STATE,
    "solver": "liblinear",
}
PARAMS[_SVC] = {"random_state": RANDOM_STATE}
PARAMS[_MLP] = {
    "random_state": RANDOM_STATE,
    "max_iter": 100000,
    "early_stopping": True,
    "learning_rate": "adaptive",
    "solver": "lbfgs",
}
PARAMS[_XGB] = {
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "max_delta_step": 1,
    "booster": "gbtree",
    "sampling_method": "gradient_based",
    # "early_stopping_rounds": 10,
}


def set_objective(trial, X_train, y_train, classifier_name):
    classifier_obj = None

    if classifier_name == _LOGISTIC:
        params = {
            "C": trial.suggest_float("C", 1e-10, 1000, log=True),
        }
        classifier_obj = LogisticRegression(
            **params,
            **PARAMS[_LOGISTIC],
        )
    elif classifier_name == _SVC:
        params = {
            "C": trial.suggest_float("C", 1e-10, 100, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 3, 5),
            "gamma": trial.suggest_categorical("gamma", ["auto", "scale"]),
        }
        classifier_obj = SVC(
            **params,
            **PARAMS[_SVC],
        )
    elif classifier_name == _MLP:
        params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [
                    (100,),
                    (50,),
                    (150,),
                    (200,),
                    (100, 50),
                    (200, 50),
                    (100, 20),
                ],
            ),
            "activation": trial.suggest_categorical(
                "activation", ["logistic", "tanh", "relu"]
            ),
        }
        classifier_obj = MLPClassifier(
            **params,
            **PARAMS[_MLP],
        )
    elif classifier_name == _XGB:
        params = {
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            # "max_leaves": trial.suggest_int("max_leaves", 0, 1000, step=100),
            "eta": trial.suggest_loguniform("eta", 1e-3, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "alpha": trial.suggest_loguniform("alpha", 1e-2, 1),
            "gamma": trial.suggest_loguniform("gamma", 1e-2, 1),
            "lambda": trial.suggest_loguniform("lambda", 1e-2, 1),
        }
        classifier_obj = xgb.XGBClassifier(
            **params,
            **PARAMS[_XGB],
        )

    if classifier_name != _XGB:
        score = cross_val_score(
            classifier_obj, X_train, y_train, n_jobs=-1, cv=5, scoring="f1"
        )

    ############################################################# HELP
    else:  # early stopping for xgb
        cv = get_iterable_cvindices(y_train)

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "validation_0-f1"
        )
        score = -cross_val_score(
            classifier_obj,
            X_train,
            y_train,
            scoring="f1",
            cv=5,
            fit_params={"callbacks": [pruning_callback]},
            n_jobs=1,
        )

    return score.mean()
