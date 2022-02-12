import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.modeling.utils import check_path


def heatmap_corr_plot(df: pd.DataFrame):
    correlation_matrix = df.corr().round(2)
    plt.figure(figsize=(16, 9))
    sns.heatmap(data=correlation_matrix, annot=True)
    check_path()
    plt.savefig("plots/correlation.png")


def loan_status_count(df: pd.DataFrame):
    plt.figure(figsize=(16, 16))
    ax = sns.countplot(x="loan_status", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # chart.set_xticklabels(rotation=45)
    check_path()
    plt.savefig("plots/loan_status_count.png")


def loan_status_grade(df: pd.DataFrame):
    plt.figure(figsize=(16, 9))
    sns.countplot(x="grade", hue="loan_status", data=df)
    check_path()
    plt.savefig("plots/loan_status_grade.png")
