# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from RossmannSalesPrediction.helpers import lr_finder
from tensorflow import keras
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
from os import O_DIRECTORY


root_path = "../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
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
embedding_fts = "store dayofweek stateholiday monthofyear dayofmonth storetype assortment promointerval".split()

to_be_encoded = "stateholiday storetype assortment promointerval".split()
to_be_scaled = "avg_store_customers avg_store_sales competitiondistance sales_prev1 sales_prev7".split()
leaveasis = "store dayofweek promo schoolholiday monthofyear dayofmonth promo2".split()

#%%

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

dimtable = {k: v for k, v in zip(embedding_fts, [xtrain_nn[col].nunique() for col in embedding_fts])}
#%%
num_input = keras.Input(shape=(xtrain_nn_num.shape[1], ))
emb_inputs = [keras.Input(shape=(1, )) for _ in embedding_fts]

emb_table = {
    'store': 10,
    'dayofweek': 6,
    'stateholiday': 2,
    'monthofyear': 6,
    'dayofmonth': 10,
    'storetype': 2,
    'assortment': 2,
    'promointerval': 2
}

emb_layers = [keras.layers.Embedding(input_dim=dimtable[col]+1, output_dim=emb_table[col])(emb_inputs[idx]) for idx, col in enumerate(embedding_fts)]

#dense1 = keras.layers.Dense(units=64, activation='relu')(num_input)
flats = [keras.layers.Flatten()(x) for x in emb_layers + [num_input]]
concat = keras.layers.Concatenate()(flats)
hidden_dense = keras.layers.Dense(units=1024, activation='sigmoid')(concat)
#hidden_dense = keras.layers.Dense(units=512, activation = 'sigmoid')(hidden_dense)
hidden_dense = keras.layers.Dense(units=512, activation='sigmoid')(hidden_dense)
out = keras.layers.Dense(units=1, activation='linear')(hidden_dense)

model: keras.Model = keras.Model(inputs=emb_inputs + [num_input], outputs=out)

#%%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

train_in = np.split(xtrain_nn_emb, xtrain_nn_emb.shape[-1], axis=1) + [xtrain_nn_num]
val_in = np.split(xval_nn_emb, xval_nn_emb.shape[-1], axis=1) + [xval_nn_num]

# -> log -> SS y values
y_ss = StandardScaler()
y_ss.fit(np.log(ytrain.values.reshape(-1, 1)))
ytrain_scaled = y_ss.transform(np.log(ytrain.values.reshape(-1, 1)))
yval_scaled = y_ss.transform(np.log(yval.values.reshape(-1, 1)))

#%%
lrf = lr_finder.LRFinder(1e-6, 1e-1)

model.fit(x=train_in, y=ytrain_scaled.flatten(), callbacks=[lrf], validation_data=(val_in, yval_scaled.flatten()), epochs=1, batch_size=256)
#%%
model.compile(optimizer=keras.optimizers.Adam(10**-4), loss='mean_squared_error', metrics=['mean_squared_error'])

hist = model.fit(x=train_in, y=ytrain_scaled.flatten(), validation_data=(val_in, yval_scaled.flatten()), epochs=5, batch_size=256)

#%%
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()

#%%
preds = model.predict(val_in)

#%%

rmspcte(yval, preds)
