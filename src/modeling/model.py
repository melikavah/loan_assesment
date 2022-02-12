import pathlib
import pickle

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


def train_dataset(
    X_train,
    y_train,
    X_test,
    y_test,
    path_to_save=".",
    learning_rate=0.1,
    n_estimators=200,
    n_jobs=-1,
    colsample_bytree=0.3,
    max_depth=5,
    alpha=10,
    random_state=42,
    scale_pos_weight=5.37,
):
    """[summary]
    Args:
        X_train (np.array)
        y_train (np.array)
        X_test (np.array)
        y_test (np.array)
        learning_rate (float, optional): [description]. Defaults to 0.1.
        n_estimators (int, optional): [description]. Defaults to 200.
        n_jobs (int, optional): [description]. Defaults to -1.
        colsample_bytree (float, optional): [description]. Defaults to 0.3.
        max_depth (int, optional): [description]. Defaults to 5.
        alpha (int, optional): [description]. Defaults to 10.
        random_state (int, optional): [description]. Defaults to 42.
    Returns:
        [type]: [description]
    """
    file = pathlib.Path(path_to_save + "/finalized_model.sav")
    if file.exists():
        model = pickle.load(open(path_to_save + "/finalized_model.sav", "rb"))
    else:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            colsample_bytree=colsample_bytree,
            max_depth=max_depth,
            alpha=alpha,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=20,
        )
        save_model(model, path_to_save=path_to_save)

    results = model.evals_result()
    return model, results


def predict_dataset(model, X_test, y_test):
    """[summary]
    Args:
        model ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
    Returns:
        [type]: [description]
    """

    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    acc = accuracy_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred)
    return y_pred, acc, f1_sc


def save_model(model: xgb.sklearn.XGBClassifier, path_to_save: str = "."):
    """[summary]

    Args:
        model (xgb.sklearn.XGBClassifier): [description]
        path_to_save (str): [Path to teh directory to save the model]
    """
    # save the model to disk
    filename = path_to_save + "/finalized_model.sav"
    pickle.dump(model, open(filename, "wb"))
