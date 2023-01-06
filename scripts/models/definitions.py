import optuna
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42

_LOGISTIC = "LogisticRegression"
_SVC = "SVC"
_MLP = "MLPClassifier"
_XGB = "XGBClassifier"
NAMES = [_LOGISTIC, _SVC, _MLP, _XGB]

SCORES = [roc_auc_score, f1_score]

PARAMS = {}
PARAMS[_LOGISTIC] = {
    "max_iter": 1000000,
    "random_state": RANDOM_STATE,
    "solver": "liblinear",
}
PARAMS[_SVC] = {"random_state": RANDOM_STATE}
PARAMS[_MLP] = {
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
    "early_stopping": True,
    "learning_rate": "adaptive",
    "solver": "lbfgs",
}
PARAMS[_XGB] = {
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    # "tree_method": "gpu_hist",
    # "gpu_id": 0,
    "max_delta_step": 1,
    "booster": "gbtree",
    "sampling_method": "gradient_based",
    "eval_metric": "auc",
    # "early_stopping_rounds": 10,
}


# returns sklearn model or dict of params
def get_model_params(trial, classifier_name):
    if classifier_name == _LOGISTIC:
        params = {
            "C": trial.suggest_float("C", 1e-10, 1000, log=True),
        }
        return LogisticRegression(
            **params,
            **PARAMS[_LOGISTIC],
        )

    if classifier_name == _SVC:
        params = {
            "C": trial.suggest_float("C", 1e-10, 100, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 3, 5),
            "gamma": trial.suggest_categorical("gamma", ["auto", "scale"]),
        }
        return SVC(
            **params,
            **PARAMS[_SVC],
        )

    if classifier_name == _MLP:
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
        return MLPClassifier(
            **params,
            **PARAMS[_MLP],
        )

    # _XGB
    return {
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "max_leaves": trial.suggest_int("max_leaves", 0, 1000, step=100),
        "eta": trial.suggest_loguniform("eta", 1e-3, 1),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "alpha": trial.suggest_float("alpha", 1e-2, 1),
        "gamma": trial.suggest_float("gamma", 1e-2, 1),
        "lambda": trial.suggest_float("lambda", 1e-2, 1),
    }


def set_objective(trial, X_train, y_train, classifier_name, d_matrix=None):
    assert classifier_name in NAMES, "Classifier not recognized"

    if classifier_name != _XGB:  # sklearn interface
        classifier_obj = get_model_params(trial, classifier_name)
        return cross_val_score(
            classifier_obj, X_train, y_train, n_jobs=-1, cv=5, scoring="roc_auc"
        ).mean()

    # xgb interface
    params = get_model_params(trial, classifier_name)
    params.update(PARAMS[_XGB])

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    num_round = params.pop("n_estimators")
    cv = xgb.cv(
        params,
        d_matrix,
        num_round,
        nfold=5,
        metrics={"auc"},
        seed=RANDOM_STATE,
        callbacks=[pruning_callback],
    )
    return cv.loc[cv.shape[0] - 1, "test-auc-mean"]
