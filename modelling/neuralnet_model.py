# %%
from os import O_DIRECTORY
import sys
import pathlib

sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
from RossmannSalesPrediction.helpers import feature_engineering
import pandas as pd
import numpy as np


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
embedding_fts = "store dayofweek stateholiday monthofyear storetype assortment promointerval".split()

to_be_encoded = "stateholiday storetype assortment promointerval".split()
to_be_scaled = "avg_store_customers avg_store_sales competitiondistance".split()
leaveasis = "store dayofweek promo schoolholiday monthofyear promo2".split()

#%%
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

oe = OrdinalEncoder()
ss = StandardScaler()

ct = ColumnTransformer([
    ('leavasis', 'passthrough', leaveasis),
    ("ordinalencode", oe, to_be_encoded),
    ("standardscale", ss, to_be_scaled),
])

ct.fit(xtrain)

#%%
xtrain_nn = pd.DataFrame(ct.transform(xtrain), columns=leaveasis + to_be_encoded + to_be_scaled)
xval_nn = pd.DataFrame(ct.transform(xval), columns=leaveasis + to_be_encoded + to_be_scaled)

#%%
# Train numeric/embedding split
xtrain_nn_num = xtrain_nn.loc[:, ~xtrain_nn.columns.isin(embedding_fts)].values
xtrain_nn_emb = xtrain_nn.loc[:, embedding_fts].values

# Validation numeric/embedding split
xval_nn_num = xval_nn.loc[:, ~xval_nn.columns.isin(embedding_fts)].values
xval_nn_emb = xval_nn.loc[:, embedding_fts].values

#%%
import tensorflow as tf
from tensorflow import keras

dimtable = {k: v for k, v in zip(embedding_fts, [xtrain_nn[col].nunique() for col in embedding_fts])}
#%%
num_input = keras.layers.Input(shape=(xtrain_nn_num.shape[1], ))
emb_input = keras.layers.Input(shape=(xtrain_nn_emb.shape[1], ))

emb_layers = [keras.layers.Embedding(input_dim=dimtable[col], output_dim=int(np.ceil(np.sqrt(dimtable[col]))))(emb_input[:, idx]) for idx, col in enumerate(embedding_fts)]

dense1 = keras.layers.Dense(units=64, activation='relu')(num_input)
#flat = keras.layers.Flatten()(emb_layers + dense1)
concat = keras.layers.Concatenate()(emb_layers + [dense1])
dense2 = keras.layers.Dense(units=64, activation = 'relu')(concat)
out = keras.layers.Dense(units=1, activation='linear')(dense2)

model: keras.Model = keras.Model(inputs=[emb_input, num_input], outputs=out)

#%%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

train_tup = (xtrain_nn_emb, xtrain_nn_num)
val_tup = (xval_nn_emb, xval_nn_num)

model.fit(x=train_tup, y=ytrain, validation_data=(val_tup, yval) ,epochs=5)