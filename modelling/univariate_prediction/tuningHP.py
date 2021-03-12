# %%
import sys
import pathlib

sys.path.append(str(pathlib.Path("..").resolve().parent.parent))
import optuna
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from RossmannSalesPrediction.helpers.evaluation import rmspe_loss
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
    .drop(["competitionopensincemonth", "competitionopensinceyear"], axis=1)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    .pipe(feature_engineering.split_date)
    .pipe(feature_engineering.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(feature_engineering.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(feature_engineering.join_store_details)
    .drop(["competitionopensincemonth", "competitionopensinceyear"], axis=1)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)


#%%
embedding_fts = "store dayofweek dayofyear stateholiday monthofyear dayofmonth storetype assortment promointerval weekofyear".split()

to_be_encoded = embedding_fts  # "stateholiday storetype assortment promointerval weekofyear".split()
to_be_scaled = "avg_store_customers avg_store_sales competitiondistance elapsed_promo_fwd elapsed_promo_backwd elapsed_schoolholiday_fwd elapsed_schoolholiday_backwd".split()
leaveasis = "promo schoolholiday promo2".split()
#leaveasis = [x for x in leaveasis if x not in embedding_fts]

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


def objective(trial):
    dimtable = {k: v for k, v in zip(embedding_fts, [xtrain_nn[col].nunique() for col in embedding_fts])}

    num_input = keras.Input(shape=(xtrain_nn_num.shape[1], ))
    emb_inputs = [keras.Input(shape=(1, )) for _ in embedding_fts]

    emb_table = {
        'store': 10,
        'dayofweek': 6,
        'dayofyear': 6,
        'stateholiday': 2,
        'monthofyear': 6,
        'weekofyear': 10,
        'dayofmonth': 10,
        'storetype': 2,
        'assortment': 2,
        'promointerval': 2,
        'elapsed_promo_fwd': 5,
        'elapsed_promo_backwd': 5,
        'elapsed_schoolholiday_fwd': 5,
        'elapsed_schoolholiday_backwd': 5,
    }

    act = trial.suggest_categorical('act_fn', ['relu', 'sigmoid', 'tanh'])
    emb_layers = [keras.layers.Embedding(input_dim=dimtable[col]+1, output_dim=emb_table[col])(emb_inputs[idx]) for idx, col in enumerate(embedding_fts)]
    # num_dense = keras.layers.Dense(units=2**trial.suggest_int('units_numdense', 4, 10), activation=act, kernel_regularizer=keras.regularizers.l2())(num_input)

    #dense1 = keras.layers.Dense(units=64, activation='relu')(num_input)
    flats = [keras.layers.Flatten()(x) for x in emb_layers + [num_input]]
    concat = keras.layers.Concatenate()(flats)

    hidden_dense = keras.layers.Dense(units=2**trial.suggest_int('units_1', 5, 15), activation=act, kernel_regularizer=keras.regularizers.l2())(concat)
    hidden_dense = keras.layers.Dropout(0.2)(hidden_dense)
    hidden_dense = keras.layers.Dense(units=2**trial.suggest_int('units_2', 5, 15), activation=act, kernel_regularizer=keras.regularizers.l2())(hidden_dense)
    hidden_dense = keras.layers.Dropout(0.2)(hidden_dense)

    out = keras.layers.Dense(units=1, activation='linear', )(hidden_dense)  # bias_initializer=keras.initializers.Constant(ytrain.mean())

    model: keras.Model = keras.Model(inputs=emb_inputs + [num_input], outputs=out)

    from tensorflow.keras.callbacks import ModelCheckpoint

    mcp = ModelCheckpoint('.modelcheckpoint', save_best_only=True, save_weights_only=True)

    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=rmspe_loss, metrics=[rmspe_loss])

    train_in = np.split(xtrain_nn_emb, xtrain_nn_emb.shape[-1], axis=1) + [xtrain_nn_num]
    val_in = np.split(xval_nn_emb, xval_nn_emb.shape[-1], axis=1) + [xval_nn_num]

    bs = 512  # 2**trial.suggest_int('bs', 6, 11)
    model.fit(x=train_in, y=ytrain.values.flatten().astype(np.float32), validation_data=(val_in, yval.values.flatten().astype(
        np.float32)), epochs=8, batch_size=bs, callbacks=[mcp, keras.callbacks.ReduceLROnPlateau(patience=2)])

    model.load_weights(".modelcheckpoint")

    preds = model.predict(val_in)
    return rmspcte(yval, preds)

study = optuna.create_study(storage="sqlite:///univariate_optuna.db", study_name="univariate_nn", load_if_exists=True, sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=20, )


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
