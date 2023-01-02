import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score

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
            "C": trial.suggest_float("C", 1e-10, 1000, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 3, 7),
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
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            # "max_leaves": trial.suggest_int("max_leaves", 0, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
        }
        classifier_obj = xgb.XGBClassifier(
            **params,
            **PARAMS[_XGB],
        )

    score = cross_val_score(
        classifier_obj, X_train, y_train, n_jobs=-1, cv=3, scoring="f1"
    )
    return score.mean()
