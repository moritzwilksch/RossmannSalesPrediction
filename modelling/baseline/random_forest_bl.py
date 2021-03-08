# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import numpy as np
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from RossmannSalesPrediction.helpers import lr_finder
from RossmannSalesPrediction.helpers.evaluation import rmspcte
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys
import pathlib

sys.path.append(str(pathlib.Path("..").resolve().parent.parent))
root_path = "../../../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)

# ADD ELAPSED
promo_elapsed = (
    train.groupby('date').mean().reset_index()
    .pipe(feature_engineering.time_elapsed, 'promo', 'forward')
    .pipe(feature_engineering.time_elapsed, 'promo', 'backward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
    [['date', 'elapsed_promo_fwd', 'elapsed_promo_backwd', 'elapsed_schoolholiday_fwd', 'elapsed_schoolholiday_backwd']]
)

train = pd.merge(train, promo_elapsed, on=["date"], how='left')


xtrain_raw, xval_raw, ytrain_raw, yval_raw = timeseries_ttsplit(train)


xtrain, ytrain = (
    xtrain_raw
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)


assert (xtrain.index == ytrain.index).all()


#%%

#%%
embedding_fts = "store dayofweek dayofyear stateholiday monthofyear dayofmonth storetype assortment promointerval weekofyear elapsed_promo_fwd elapsed_promo_backwd elapsed_schoolholiday_fwd elapsed_schoolholiday_backwd".split()

xtrain[embedding_fts] = xtrain[embedding_fts].astype('category')
xval[embedding_fts] = xval[embedding_fts].astype('category')

for col in embedding_fts:
    xtrain[col] = xtrain[col].cat.codes
    xval[col] = xval[col].cat.codes
#%%

rfr = RandomForestRegressor(n_jobs=-1)
#rfr = LinearRegression(n_jobs=-1)
rfr.fit(xtrain, ytrain)

#%%
preds = rfr.predict(xval)

#%%
rmspcte(yval, preds)

#%%
# REFIT WHOLE DATASET
xall = pd.concat((xtrain, xval), axis=0)
yall = pd.concat((ytrain, yval), axis=0)

rfr.fit(xall, yall)

#%%
# TEST SUBMISSION RANDOM FOREST
if True:
    test_raw = pd.read_csv(root_path + "data/test.csv").drop('Id', axis=1).reset_index(drop=True)
    test = fix_df(test_raw.copy())

    promo_elapsed = (
        test.groupby('date').mean().reset_index()
        .pipe(feature_engineering.time_elapsed, 'promo', 'forward')
        .pipe(feature_engineering.time_elapsed, 'promo', 'backward')
        .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
        .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
        [['date', 'elapsed_promo_fwd', 'elapsed_promo_backwd', 'elapsed_schoolholiday_fwd', 'elapsed_schoolholiday_backwd']]
    )

    test = pd.merge(test, promo_elapsed, on=["date"], how='left')

    xall_raw = pd.concat((xtrain_raw, xval_raw), axis=0)
    yall_raw = pd.concat((ytrain_raw, yval_raw), axis=0)

    xtest = (
        test
        .pipe(feature_engineering.split_date)
        .pipe(feature_engineering.add_avg_customers_per_store, train_data=xall_raw)
        .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xall_raw, ytrain=yall_raw)
        .pipe(feature_engineering.join_store_details)
        .drop(['date', 'open'], axis=1)
        .rename({'customers': 'avg_store_customers'}, axis=1)
        [xval.columns]
    )

    for col in embedding_fts:
        xtest[col] = xtest[col].astype('category')
        xtest[col] = xtest[col].cat.codes

    print(xtest.info())

    preds = rfr.predict(xtest)
    submission = pd.DataFrame({'Id': range(1, len(preds)+1), 'sales': preds.flatten()}).set_index('Id')
    submission
    submission.to_csv("random_forest_submission.csv")
