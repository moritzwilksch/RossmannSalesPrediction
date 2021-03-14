# %%
import sys
import pathlib
from typing import Union

sys.path.append(str(pathlib.Path("..").resolve().parent.parent))

from tensorflow.keras.callbacks import ModelCheckpoint
from RossmannSalesPrediction.helpers.evaluation import rmspe_loss
from RossmannSalesPrediction.helpers import feature_engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
import seaborn as sns



root_path = "RossmannSalesPrediction/"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
X = train.groupby('date').mean().drop(['store', 'sales', 'customers'], axis=1).reset_index()
y = pd.pivot_table(train, index='date', columns='store', values='sales').fillna(0).values

cutoff = int(len(y) * 0.8)

xtrain_raw = X.iloc[:cutoff]
ytrain = y[:cutoff]

xval_raw = X.iloc[cutoff:]
yval = y[cutoff:]

#%%
xtrain_raw = (
    xtrain_raw
    .pipe(feature_engineering.time_elapsed, 'promo', 'forward')
    .pipe(feature_engineering.time_elapsed, 'promo', 'backward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
)

xval_raw = (
    xval_raw
    .pipe(feature_engineering.time_elapsed, 'promo', 'forward')
    .pipe(feature_engineering.time_elapsed, 'promo', 'backward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'forward')
    .pipe(feature_engineering.time_elapsed, 'schoolholiday', 'backward')
)

#%%
xtrain = feature_engineering.split_date(xtrain_raw).drop('date', axis=1)
xval = feature_engineering.split_date(xval_raw).drop('date', axis=1)


#%%
embedding_fts = "dayofweek monthofyear dayofmonth weekofyear dayofyear promo".split()
to_scale = ['elapsed_promo_fwd',
            'elapsed_promo_backwd', 'elapsed_schoolholiday_fwd',
            'elapsed_schoolholiday_backwd']
#%%
ss = StandardScaler()
ss.fit(xtrain[to_scale])
xtrain[to_scale] = ss.transform(xtrain[to_scale])
xval[to_scale] = ss.transform(xval[to_scale])


#%%
xtrain_nn = xtrain.copy()
xval_nn = xval.copy()

xtrain[embedding_fts] = xtrain[embedding_fts].astype('int64')
xval[embedding_fts] = xval[embedding_fts].astype('int64')


train_wasopen = (xtrain.open > 0.05)
val_wasopen = (xval.open > 0.05)

xtrain = xtrain[train_wasopen]
xval = xval[val_wasopen]

ytrain = ytrain[train_wasopen, :]
yval = yval[val_wasopen, :]


trainset = tf.data.Dataset.from_tensor_slices((dict(xtrain), ytrain.astype(np.float32))).batch(32)
valset = tf.data.Dataset.from_tensor_slices((dict(xval), yval.astype(np.float32))).batch(len(yval))


feature_cols = []
ftl_inputs = {}

emb_table = {
    'dayofweek': 10,  # 6,
    'monthofyear': 15,  # 6,
    'dayofmonth': 10,
    'weekofyear': 25,
    'dayofyear': 5,
    'promo': 2
}


for col in xtrain.columns:
    if col in embedding_fts:
        cat = tf.feature_column.categorical_column_with_vocabulary_list(col, vocabulary_list=xtrain[col].unique())
        emb = tf.feature_column.embedding_column(cat, dimension=emb_table[col])
        feature_cols.append(emb)
        ftl_inputs[col] = tf.keras.Input(shape=(1,), dtype=tf.int8, name=col)

    else:
        num = tf.feature_column.numeric_column(col)
        feature_cols.append(num)
        ftl_inputs[col] = tf.keras.Input(shape=(1,), dtype=tf.float32, name=col)


ftcols_in = tf.keras.layers.DenseFeatures(feature_cols)(ftl_inputs)


#flats = [keras.layers.Flatten()(x) for x in emb_layers + [num_input]]
# concat = keras.layers.Concatenate()(flats)
# hidden_dense = keras.layers.Dense(units=512, activation='relu',)(concat)

hidden_dense = keras.layers.Dense(units=512, activation='relu',)(ftcols_in)
hidden_dense = keras.layers.Dense(units=512, activation='relu',)(hidden_dense)
out = keras.layers.Dense(units=1115, activation='linear')(hidden_dense)

#model: keras.Model = keras.Model(inputs=emb_inputs + [num_input], outputs=out)
model: keras.Model = keras.Model(inputs=[v for v in ftl_inputs.values()], outputs=out)

#%%

mcp = ModelCheckpoint('.modelcheckpoint', save_best_only=True, save_weights_only=True)
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss=rmspe_loss, metrics=[rmspe_loss])

#%%
hist = model.fit(trainset, validation_data=valset, epochs=20, ) # callbacks=[mcp])

model.load_weights(".modelcheckpoint")


#%%

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()
plt.show()

preds = model.predict(valset)
#preds = y_ss.inverse_transform(preds)


print(rmspcte(yval, preds))

sns.histplot(preds, color='orange')
sns.histplot(yval)
plt.show()
