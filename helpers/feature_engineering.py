# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
import pandas as pd
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df

from numpy.lib.index_tricks import IndexExpression


root_path = "../../"

# %%


def split_date(data: pd.DataFrame) -> pd.DataFrame:
    data['monthofyear'] = data.date.dt.month
    data['dayofmonth'] = data.date.dt.day
    data['dayofyear'] = data.date.dt.dayofyear
    data['weekofyear'] = data.date.dt.isocalendar().week.astype('category')
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


def join_store_details(data: pd.DataFrame) -> pd.DataFrame:
    catcols = "storetype assortment promointerval".split()

    store = pd.read_csv(root_path + "data/store.csv", index_col='Store')
    store.columns = [col.lower() for col in store.columns]

    store[catcols] = store[catcols].astype('category')

    # add no regular promo
    store['promointerval'].cat.add_categories('NotRegular', inplace=True)
    store['promointerval'] = store['promointerval'].fillna('NotRegular')

    store = store.drop("competitionopensincemonth competitionopensinceyear promo2sinceweek promo2sinceyear".split(), axis=1)

    # impute competition distance to median of train set
    store['competitiondistance'] = store['competitiondistance'].fillna(2330.0)

    return pd.merge(data, store, how='left', left_on='store', right_index=True)


def add_time_lag(x: pd.DataFrame, y: pd.Series, lag: int) -> pd.DataFrame:
    train = pd.concat((x, y), axis=1)
    train.set_index(x.index)
    sales_lookup = train[['store', 'date', 'sales']].copy()
    sales_lookup = sales_lookup.assign(joindate=sales_lookup.date + pd.Timedelta(lag, 'd'))

    train_aug = (
        pd
        .merge(train, sales_lookup.drop(['date'], axis=1), how='left', left_on=['store', 'date'], right_on=['store', 'joindate'])
        .rename({'sales_y': f'sales_prev{lag}'}, axis=1)
        .rename({'sales_x': 'sales'}, axis=1)
        .drop(['joindate', 'sales'], axis=1)
        .set_index(x.index)
    )

    return train_aug

def add_moving_avg(x: pd.DataFrame, y: pd.Series, size: int) -> pd.DataFrame:
        train = pd.concat((x, y), axis=1)
        #train.set_index(x.index)
        join_table = (
            train
            #.sort_values(['store', 'date'])
            .groupby('store')[['sales', 'date']]
            .rolling(size, min_periods=1)
            .mean()
            .reset_index()
            .set_index('level_1')
            .rename({'sales': f'ma_{size}'}, axis=1)
            )
        return (
            pd.merge(x, join_table, left_index=True, right_index=True, how='left')
            .drop('store_y', axis=1)
            .rename({'store_x': 'store'}, axis=1)
            )


def time_elapsed(data: pd.DataFrame, column: str, mode: str) -> pd.DataFrame:
    """- mode: 'forward' or 'backward'"""
    data = data.copy()
    event_dates = data.loc[data[column] != 0].date

    result = []
    for row in data.itertuples():
        deltas = (row.date - event_dates)
        if mode == 'forward':
            result.append(deltas[deltas <= pd.to_timedelta(0, 'D')].max())
        elif mode == 'backward':
            result.append(deltas[deltas >= pd.to_timedelta(0, 'D')].min())

    fill_value = pd.Series(result).min() if mode == 'forward' else pd.Series(result).max()

    name = 'fwd' if mode == 'forward' else 'backwd'
    data[f"elapsed_{column}_{name}"] = result
    data[f"elapsed_{column}_{name}"] = data[f"elapsed_{column}_{name}"].fillna(pd.to_timedelta(fill_value, 'D'))
    data[f"elapsed_{column}_{name}"] = data[f"elapsed_{column}_{name}"].dt.days

    return data





"""#%%
root_path = "../"
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
add_moving_avg(train.drop('sales', axis=1), train.sales, 7)"""
