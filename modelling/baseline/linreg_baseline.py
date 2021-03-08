# %%
import sys
import pathlib

sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from helpers.evaluation import rmspe_loss
import numpy as np
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from RossmannSalesPrediction.helpers import lr_finder
from RossmannSalesPrediction.helpers.evaluation import rmspcte
import seaborn as sns


root_path = "../../"

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
from sklearn.linear_model import LinearRegression

#%%
embedding_fts = "store dayofweek dayofyear stateholiday monthofyear dayofmonth storetype assortment promointerval weekofyear elapsed_promo_fwd elapsed_promo_backwd elapsed_schoolholiday_fwd elapsed_schoolholiday_backwd".split()

xtrain[embedding_fts] = xtrain[embedding_fts].astype('category')
xval[embedding_fts] = xval[embedding_fts].astype('category')
for ft in embedding_fts:
    xtrain[ft] = xtrain[ft].cat.codes
    xval[ft] = xval[ft].cat.codes


#%%
#from sklearn.feature_extraction import FeatureHasher
#fh = FeatureHasher(n_features=100)
#fh.fit_transform(xtrain[embedding_fts]) 


#%%
lr = LinearRegression()
lr.fit(xtrain, ytrain)

#%%
rmspcte(yval, lr.predict(xval))






#%%
if False:
    # TEST SUBMISSION

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

    test = (
        test
        .pipe(feature_engineering.split_date)
        .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
        .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
        .pipe(feature_engineering.join_store_details)
        .drop(['date', 'open'], axis=1)
        .rename({'customers': 'avg_store_customers'}, axis=1)
    )

    embedding_fts = "store dayofweek dayofyear stateholiday monthofyear dayofmonth storetype assortment promointerval weekofyear".split()

    to_be_encoded = embedding_fts
    to_be_scaled = "avg_store_customers avg_store_sales competitiondistance elapsed_promo_fwd elapsed_promo_backwd elapsed_schoolholiday_fwd elapsed_schoolholiday_backwd".split()
    leaveasis = "promo schoolholiday promo2".split()

    #%%
    xtest_nn = pd.DataFrame(ct.transform(test), columns=leaveasis + to_be_encoded + to_be_scaled)

    xtest_nn_num = xtest_nn.loc[:, ~xtest_nn.columns.isin(embedding_fts)].values.astype(np.float64)
    xtest_nn_emb = xtest_nn.loc[:, embedding_fts].values.astype(np.long)
    test_in = np.split(xtest_nn_emb, xtest_nn_emb.shape[-1], axis=1) + [xtest_nn_num]

    preds = model.predict(test_in)

    output = pd.DataFrame({'Id': list(range(1, 41089)), 'sales': preds.flatten()})
    output.to_csv("correct_univar_unopt.csv", index=False)
