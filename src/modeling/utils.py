import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix


def cross_validation_report(X_train, y_train):
    """[summary]
    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
    """
    DM_train = xgb.DMatrix(data=X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "colsample_bytree": 0.3,
        "learning_rate": 0.1,
        "max_depth": 5,
        "alpha": 10,
    }
    cv_results = xgb.cv(
        nfold=3,
        dtrain=DM_train,
        params=params,
        num_boost_round=50,
        early_stopping_rounds=10,
        as_pandas=True,
        seed=123,
        metrics="logloss",
    )
    check_path()
    cv_results.to_csv("plots/cross_validation_report.csv", index=False)
    return params


def xgboost_tree_plot(X_train, y_train, params: dict):
    """This fucntion plots the xgboost tree used for classification

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
        params (dict): [The optimal xgboost parameters]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Convert the dataset into an optimized data structure called Dmatrix
        # that XGBoost supports and gives it acclaimed performance and efficiency gains.
        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        xgb_reg_1 = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

        xgb.plot_tree(xgb_reg_1, num_trees=0)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        check_path()
        fig.savefig("plots/xgboost_tree.png")


def feature_importance_plot(
    df: pd.DataFrame, model: xgb.sklearn.XGBClassifier, n: int = 10
):
    """This fucntion plots the n most important features in classification

    Args:
        df (pd.DataFrame): [description]
        model (xgb.sklearn.XGBClassifier): [description]
        n (int, optional): [description]. Defaults to 10.
    """
    df_ = df.drop(columns=["loan_status"])
    feature_names = df_.columns.values
    plt.figure(figsize=(20, 20))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(
        feature_names[sorted_idx[-n:]],
        model.feature_importances_[sorted_idx[-n:]],
    )
    plt.xlabel("Xgboost Feature Importance")
    check_path()
    plt.savefig("plots/feature_importance_plot.png")


def optimal_num_tree_graph(model, results):
    """[summary]
    Args:
        model ([type]): [description]
        results ([type]): [description]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plt.figure(figsize=(10, 7))
        plt.plot(results["validation_0"]["logloss"], label="Training loss", marker="*")
        plt.plot(results["validation_1"]["logloss"], label="Validation loss")
        plt.axvline(
            x=model.best_ntree_limit,
            ymin=0,
            ymax=14,
            color="gray",
            label="Optimal tree number",
        )
        plt.xlabel("Number of Tree")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        check_path()
        plt.savefig("plots/optimal_num_tree.png")


def check_path():
    if not os.path.exists("plots/"):
        os.mkdir("plots/")


def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 9))
    sns.heatmap(conf_matrix / np.sum(conf_matrix), annot=True, fmt=".2%", cmap="Blues")
    check_path()
    plt.savefig("plots/confusion_matrix.png")
