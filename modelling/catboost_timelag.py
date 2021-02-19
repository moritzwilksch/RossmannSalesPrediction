# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from catboost import CatBoostRegressor
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model



root_path = "../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)

#%%


#%%
# 1114 NAs for Jan 1st 2013, 1 NA for Jan 2nd 2013, 180 NAs for Jan 1st 2015
#train_aug.loc[train_aug.isna().any(axis=1)].groupby('date').count()

#%%
xtrain_raw, xval_raw, ytrain_raw, yval_raw = timeseries_ttsplit(train)


xtrain, ytrain = (
    xtrain_raw
    .pipe(feature_engineering.add_time_lag, lag=1, y=ytrain_raw)
    .pipe(feature_engineering.add_time_lag, lag=7, y=ytrain_raw)
    .dropna()
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    .pipe(feature_engineering.add_time_lag, lag=1, y=yval_raw)
    .pipe(feature_engineering.add_time_lag, lag=7, y=yval_raw)
    .dropna()
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)

#%%
xtrain
#%%
cbr = CatBoostRegressor(iterations=150, verbose=10, loss_function='RMSE')
cbr.fit(xtrain, ytrain, eval_set=(xval, yval), cat_features="store dayofweek stateholiday monthofyear storetype assortment promointerval".split())

#%%

preds = cbr.predict(xval)
print(rmspcte(yval, preds))
