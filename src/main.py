import click
import pandas as pd

from src.data_preprocessing.eda import (
    heatmap_corr_plot,
    loan_status_count,
    loan_status_grade,
)

from .data_preprocessing.preprocessing import preprocessing, remove_redundant_target
from .modeling.model import predict_dataset, train_dataset
from .modeling.utils import (
    cross_validation_report,
    feature_importance_plot,
    optimal_num_tree_graph,
    plot_confusion_matrix,
    xgboost_tree_plot,
)


@click.command()
@click.option(
    "-p", "--path-dataset", prompt=False, default="data/accepted_2007_to_2018Q4.csv"
)
def main(path_dataset):
    # Reading the dataset using pandas
    df = pd.read_csv(path_dataset)
    ## --------- EDA ---------------
    heatmap_corr_plot(df)
    _df = remove_redundant_target(df)
    loan_status_count(_df)
    loan_status_grade(_df)
    ## --------- preprocessing ---------------
    df, X_train, X_test, y_train, y_test = preprocessing(df)

    ## --------- training and validation
    model, results = train_dataset(X_train, y_train, X_test, y_test)
    y_pred, acc, f1_sc = predict_dataset(model, X_test, y_test)
    print(f"The accuracy of the model is: {acc}")
    print(f"The F1-Score is {f1_sc}")
    plot_confusion_matrix(y_test, y_pred)
    optimal_num_tree_graph(model, results)
    params = cross_validation_report(X_train, y_train)
    xgboost_tree_plot(X_train, y_train, params)
    feature_importance_plot(df, model)


if __name__ == "__main__":
    main()
