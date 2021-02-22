# %%
import sys
import pathlib

sys.path.append(str(pathlib.Path("..").resolve().parent.parent))
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model


root_path = "../../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
xtrain_raw, xval_raw, ytrain_raw, yval_raw = timeseries_ttsplit(train)


xtrain, ytrain = (
    xtrain_raw
    #.pipe(feature_engineering.time_elapsed, 'promo', 'forward')
    #.pipe(feature_engineering.time_elapsed, 'promo', 'backward')
    #.pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
    #.pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    #.pipe(feature_engineering.time_elapsed, 'promo', 'forward')
    #.pipe(feature_engineering.time_elapsed, 'promo', 'backward')
    #.pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
    #.pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)

#%%
xtrain

#%%
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_jobs=-1)
rfr.fit(xtrain, ytrain)