#%%
import pandas as pd
from typing import Tuple
root_path = "../"


def fix_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to lower case.
    Fixes `object` data types
    """
    data = data.rename({col: col.lower() for col in data.columns}, axis=1)
    data["date"] = data['date'].astype('datetime64')
    data['stateholiday'] = (
        data['stateholiday']
        .map({0: 'no', '0': 'no', 'a': 'public', 'b': 'easter', 'c': 'xmas'})
        .astype('category')
    )

    return data


def timeseries_ttsplit(data: pd.DataFrame, train_pct=0.8) -> pd.DataFrame:
    """
    Splits `data` into train & test using *first* `train_pct` percent *of days* as train data.
    - Rounds to a full day, everything before is train, after is test.
    """
    data: pd.DataFrame = data.copy()

    n_days_total = (data.date.max() - data.date.min()).days
    n_days_train = int(train_pct * n_days_total)
    thresh_date = data.date.min() + pd.Timedelta(n_days_train, unit='d')

    train_mask = (data.date <= thresh_date)

    xtrain = data.drop('sales', axis=1).loc[train_mask, :]
    xval = data.drop('sales', axis=1).loc[~train_mask, :]
    ytrain = data.loc[train_mask, 'sales']
    yval = data.loc[~train_mask, 'sales']

    return xtrain, xval, ytrain, yval


def prep_for_model(x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - Removes `customers` as it is not known @ inference.
    - Removes exact `date` to prevent date memorization.
    - Removes samples with open == 0
    - Removes samples with sales == 0
    """
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    open_mask = (x.open != 0)
    sales_mask = (y > 1e-9)

    mask = (open_mask & sales_mask).values

    x = x.loc[mask, :]
    x = x.drop(['customers_x', 'date', 'open'], axis=1)
    y = y[mask]
    return x.reset_index(drop=True), y.reset_index(drop=True)
