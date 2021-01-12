# %%
import sys
import pathlib

from numpy.lib.index_tricks import IndexExpression
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df
import pandas as pd


root_path = "../"

# %%


def add_monthofyear(data: pd.DataFrame) -> pd.DataFrame:
    data['monthofyear'] = data.date.dt.month
    return data


def add_avg_customers_per_store(data: pd.DataFrame, train_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean customers per store on `train_data` and joins to `data` on store id"""
    per_store = train_data.groupby('store')['customers'].mean().to_frame()
    return pd.merge(data, per_store, left_on='store', right_index=True).rename({'customers_y': 'avg_store_customers'}, axis=1)


def add_avg_sales_per_store(data: pd.DataFrame, xtrain: pd.DataFrame, ytrain: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean sales per store on `xtrain` and `ytrain` and joins to `data` on store id"""

    if not all(xtrain.index == ytrain.index):
        raise Exception("Index mismatch between xtrain & ytrain")

    xandy = pd.concat([xtrain, ytrain], axis=1)
    per_store = xandy.groupby('store')['sales'].mean().to_frame().rename({'sales': 'avg_store_sales'}, axis=1)

    return pd.merge(data, per_store, left_on='store', right_index=True)


if __name__ == "__main__":
    train = pd.read_csv(root_path + 'data/train.csv')
    train = fix_df(train)
    xtrain, xval, ytrain, yval = timeseries_ttsplit(train)

    print(
        xtrain
        .pipe(add_avg_customers_per_store, train_data=xtrain)
        .pipe(add_avg_sales_per_store, xtrain=xtrain, ytrain=ytrain)
    )

    print(
        xval
        .pipe(add_avg_customers_per_store, train_data=xtrain)
        .pipe(add_avg_sales_per_store, xtrain=xtrain, ytrain=ytrain)
    )