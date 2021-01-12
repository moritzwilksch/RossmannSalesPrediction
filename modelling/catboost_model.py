# %%
import sys
import pathlib

from numpy.lib.index_tricks import IndexExpression
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
from RossmannSalesPrediction.helpers import feature_engineering
import pandas as pd


root_path = "../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
xtrain_raw, xval_raw, ytrain_raw, yval_raw = timeseries_ttsplit(train)


xtrain, ytrain = (
    xtrain_raw
    .pipe(feature_engineering.add_monthofyear)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    .pipe(feature_engineering.add_monthofyear)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)

#%%
xtrain
#%%
from catboost import CatBoostRegressor
cbr = CatBoostRegressor(iterations=50, verbose=10, loss_function='RMSE')
cbr.fit(xtrain, ytrain, eval_set=(xval, yval), cat_features="store dayofweek stateholiday monthofyear storetype assortment promointerval".split())

#%%
from RossmannSalesPrediction.helpers.evaluation import rmspcte

preds = cbr.predict(xval)
print(rmspcte(yval, preds))