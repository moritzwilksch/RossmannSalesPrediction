# %%
import seaborn as sns
import sys
import pathlib
from typing import Union

sys.path.append(str(pathlib.Path("..").resolve().parent.parent))
from RossmannSalesPrediction.helpers.dataprep import timeseries_ttsplit, fix_df, prep_for_model
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from tensorflow import keras
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RossmannSalesPrediction.helpers import feature_engineering
from helpers.evaluation import rmspe_loss


root_path = "../../"

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
# -> log -> SS y values
#y_ss = StandardScaler()
#y_ss.fit(ytrain)
#ytrain_scaled = y_ss.transform(ytrain)
#yval_scaled = y_ss.transform(yval)


#%%
embedding_fts = "dayofweek monthofyear dayofmonth weekofyear".split()
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

#%%
# Train numeric/embedding split
xtrain_nn_num = xtrain_nn.loc[:, ~xtrain_nn.columns.isin(embedding_fts)].values.astype(np.float64)
xtrain_nn_emb = xtrain_nn.loc[:, embedding_fts].values.astype(np.long)

# Validation numeric/embedding split
xval_nn_num = xval_nn.loc[:, ~xval_nn.columns.isin(embedding_fts)].values.astype(np.float64)
xval_nn_emb = xval_nn.loc[:, embedding_fts].values.astype(np.long)

#%%

dimtable = {k: v for k, v in zip(embedding_fts, [xtrain_nn[col].nunique() for col in embedding_fts])}
#%%
num_input = keras.Input(shape=(xtrain_nn_num.shape[1], ))
emb_inputs = [keras.Input(shape=(1, )) for _ in embedding_fts]

emb_table = {
    'dayofweek': 6,
    'monthofyear': 6,
    'dayofmonth': 10,
}

#%%
import optuna

def objective(trial):
    emb_layers = [keras.layers.Embedding(input_dim=dimtable[col]+1, output_dim=emb_table[col])(emb_inputs[idx]) for idx, col in enumerate(embedding_fts)]

    #dense1 = keras.layers.Dense(units=64, activation='relu')(num_input)
    flats = [keras.layers.Flatten()(x) for x in emb_layers + [num_input]]
    concat = keras.layers.Concatenate()(flats)

    act = trial.suggest_categorical('act_fn', ['relu', 'sigmoid', 'tanh'])
    hidden_dense = keras.layers.Dense(units=trial.suggest_int('units_1', 128, 4096, 512), activation=act, kernel_regularizer=keras.regularizers.l2())(concat)
    hidden_dense = keras.layers.Dense(units=trial.suggest_int('units_2', 128, 4096, 512), activation=act, kernel_regularizer=keras.regularizers.l2())(hidden_dense)
    out = keras.layers.Dense(units=1115, activation='linear')(hidden_dense)

    model: keras.Model = keras.Model(inputs=emb_inputs + [num_input], outputs=out)

    from tensorflow.keras.callbacks import ModelCheckpoint

    mcp = ModelCheckpoint('.modelcheckpoint', save_best_only=True, save_weights_only=True)


    model.compile(optimizer='adam', loss=rmspe_loss, metrics=[rmspe_loss])

    train_in = np.split(xtrain_nn_emb, xtrain_nn_emb.shape[-1], axis=1) + [xtrain_nn_num]
    val_in = np.split(xval_nn_emb, xval_nn_emb.shape[-1], axis=1) + [xval_nn_num]

    model.fit(x=train_in, y=ytrain_scaled, validation_data=(val_in, yval_scaled), epochs=25, batch_size=trial.suggest_int('bs', 8, 128, 8), callbacks=[mcp])

    model.load_weights(".modelcheckpoint")

    preds = y_ss.inverse_transform(model.predict(val_in))
    return rmspcte(yval, preds)

study = optuna.create_study(storage="sqlite:///multivar_optuna.db", study_name="multivariate_nn")
study.optimize(objective, n_trials=55, )

#%%

























#%%
emb_table = {
    'dayofweek': 10, #6,
    'monthofyear': 15, #6,
    'dayofmonth': 10,
    'weekofyear': 25
}
emb_layers = [keras.layers.Embedding(input_dim=dimtable[col]+1, output_dim=emb_table[col])(emb_inputs[idx]) for idx, col in enumerate(embedding_fts)]

#dense1 = keras.layers.Dense(units=64, activation='relu')(num_input)
flats = [keras.layers.Flatten()(x) for x in emb_layers + [num_input]]
concat = keras.layers.Concatenate()(flats)
hidden_dense = keras.layers.Dense(units=512, activation='relu',)(concat)
#hidden_dense = keras.layers.Dense(units=4024, activation='relu',)(hidden_dense)
hidden_dense = keras.layers.Dense(units=4664, activation='relu',)(hidden_dense)
out = keras.layers.Dense(units=1115, activation='linear')(hidden_dense)

model: keras.Model = keras.Model(inputs=emb_inputs + [num_input], outputs=out)

#%%
from tensorflow.keras.callbacks import ModelCheckpoint

mcp = ModelCheckpoint('.modelcheckpoint', save_best_only=True, save_weights_only=True)
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss=rmspe_loss, metrics=[rmspe_loss])

train_in = np.split(xtrain_nn_emb, xtrain_nn_emb.shape[-1], axis=1) + [xtrain_nn_num]
val_in = np.split(xval_nn_emb, xval_nn_emb.shape[-1], axis=1) + [xval_nn_num]

#%%
hist = model.fit(x=train_in, y=ytrain, validation_data=(val_in, yval), epochs=10, batch_size=32, callbacks=[mcp])

model.load_weights(".modelcheckpoint")

#%%

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()
plt.show()

preds = model.predict(val_in)
#preds = y_ss.inverse_transform(preds)


print(rmspcte(yval, preds))

sns.histplot(preds, color='orange')
sns.histplot(yval)
plt.show()


#%%

# TEST SUBMISSION

test_raw = pd.read_csv(root_path + "data/test.csv").drop('Id', axis=1).reset_index(drop=True)
test = fix_df(test_raw.copy())
test = test.groupby('date').mean().drop(['store'], axis=1).reset_index()
test = feature_engineering.split_date(test).drop('date', axis=1)

xtest_nn = test.copy()
xtest_nn_num = xtest_nn.loc[:, ~xtest_nn.columns.isin(embedding_fts)].values.astype(np.float64)
xtest_nn_emb = xtest_nn.loc[:, embedding_fts].values
test_in = np.split(xtest_nn_emb, xtest_nn_emb.shape[-1], axis=1) + [xtest_nn_num]

preds = y_ss.inverse_transform(model.predict(test_in))

preds_df = pd.melt(pd.DataFrame(preds, index=test_raw.Date.unique()), var_name='Store', value_name='Sales', ignore_index=False).reset_index().rename({'index': 'Date'}, axis=1)
preds_df['Store'] += 1

output = pd.merge(test_raw, preds_df, on=['Date', 'Store'], how='left')
output = output.set_index(output.index+1)
output['Sales'].to_csv(root_path + "data/multivariate_nn_out.csv", index_label='Id')
