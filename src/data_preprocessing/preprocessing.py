from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def drop_waste_columns(df: pd.DataFrame):
    _df = df.copy()
    _df.drop(
        columns=[
            "id",
            "url",
        ],
        inplace=True,
    )
    return _df


def columns_to_keep(df: pd.DataFrame) -> List:
    """keep columns with less than 50 % Nans

    Args:
        df (pd.DataFrame)

    Returns:
        cols_to_keep[List]: [The list of columns with less than 50 % Nans]
    """

    cols_to_keep = [col for col in df.columns if df[col].isna().sum() < df.shape[0] / 2]

    return cols_to_keep


def filter_on_issue_date(df: pd.DataFrame, issue_date: int = 2016) -> pd.DataFrame:
    issue_date = str(issue_date)
    df["issue_d"] = df["issue_d"].apply(lambda x: x.split()[-1][-4:])

    return df[df["issue_d"] == issue_date]


def remove_redundant_target(df: pd.DataFrame) -> pd.DataFrame:
    df[
        (df["loan_status"] == "Fully Paid")
        | (df["loan_status"] == "Charged Off")
        | (df["loan_status"] == "Current")
    ]
    return df


def map_chargeoff_to_one(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_status"] = df["loan_status"].apply(
        lambda x: 1 if x == "Charged Off" else 0
    )
    return df


def numerical_cols(df: pd.DataFrame) -> List:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        List: [description]
    """
    numeric_cols = [
        col
        for col in df.columns
        if df[col].dtype == "float64" or df[col].dtype == "int64"
    ]
    return numeric_cols


def categorical_cols(df: pd.DataFrame) -> List:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        List: [description]
    """
    cols_list = df.columns
    numeric_cols = numerical_cols(df)
    return list(set(cols_list) - set(numeric_cols))


def drop_cols_more_than_certain_cat(df: pd.DataFrame) -> List:
    cols_to_drop = [col for col in df.columns if len(df[col].unique()) > 11]
    return cols_to_drop


def drop_cols_with_one_cat(df: pd.DataFrame) -> List:
    cols_to_drop = [col for col in df.columns if len(df[col].unique()) == 1]

    return cols_to_drop


def getting_dummies(df: pd.DataFrame, categorical_cols: List):
    dummies = pd.get_dummies(df[categorical_cols], drop_first=True)
    df.drop(columns=categorical_cols, axis=1)
    df = pd.concat([df, dummies], axis=1)
    return df


def scaling_splitting(df: pd.DataFrame):
    """This fucntion splits the datset into training set and test set
    and scales the dataset

    Args:
        df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    X = df.drop("loan_status", axis=1).values
    y = df["loan_status"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=101
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def preprocessing(df: pd.DataFrame):
    """This fucntion puts all the preprocessing steps altogether

    Args:
        df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    df = drop_waste_columns(df)

    cols_to_keep = columns_to_keep(df)
    df = df[cols_to_keep]

    df.dropna(axis=0, inplace=True)
    df = filter_on_issue_date(df, issue_date=2016)
    df.drop(columns=["issue_d"], inplace=True)
    df = remove_redundant_target(df)
    df = map_chargeoff_to_one(df)
    catego_cols = categorical_cols(df)
    cols_to_drop = drop_cols_more_than_certain_cat(df[catego_cols])
    df.drop(columns=cols_to_drop, inplace=True)
    cols_with_one_cat = drop_cols_with_one_cat(df)

    df.drop(columns=cols_with_one_cat, inplace=True)
    cat_cols_final = categorical_cols(df)
    df = getting_dummies(df, cat_cols_final)
    df.drop(columns=cat_cols_final, inplace=True)

    X_train, X_test, y_train, y_test = scaling_splitting(df)
    return df, X_train, X_test, y_train, y_test
