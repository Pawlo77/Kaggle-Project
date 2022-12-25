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

from .definitions import set_objective, _LOGISTIC, _SVC, _MLP, _XGB, SCORES
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


def find_model(study_name="banking"):
    X_train = load_data("X_train_f.csv").to_numpy()
    y_train = load_data("y_train_f.csv").to_numpy().ravel()
    X_test = load_data("X_test_f.csv").to_numpy()
    y_test = load_data("y_test_f.csv").to_numpy().ravel()

    # list of pais model object, model training score
    models = []
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name=study_name,
    )

    for classifier_name in [_LOGISTIC, _SVC, _MLP, _XGB]:
        objective = partial(
            set_objective,
            X_train=X_train,
            y_train=y_train,
            classifier_name=classifier_name,
        )
        study.optimize(objective, n_trials=100, n_jobs=-1)

        if classifier_name == _LOGISTIC:
            model = LogisticRegression
        elif classifier_name == _SVC:
            model = SVC
        elif classifier_name == _MLP:
            model = MLPClassifier
        elif classifier_name == _XGB:
            model = xgb.XGBClassifier

        model = model(**study.best_params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        scores_train = get_scores(y_train, y_pred_train)
        scores_test = get_scores(y_test, y_pred_test)
        print_scores(model.__class__, scores_train, scores_test)
        models.append((model, scores_test))

    models.sort(key=lambda x: x[1], reverse=True)
    save_object(models[0], "final_model.pkl")
