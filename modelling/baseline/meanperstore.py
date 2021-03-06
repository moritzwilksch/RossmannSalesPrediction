#%%
import sys
import pathlib
sys.path.append(str(pathlib.Path("..").resolve().parent.parent))
from rich.console import Console
import numpy as np
import pandas as pd
import RossmannSalesPrediction.helpers.feature_engineering as fe
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from RossmannSalesPrediction.helpers.dataprep import fix_df, timeseries_ttsplit, prep_for_model



c = Console(highlight=False)

root_path = "../../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)

xtrain_raw, xval_raw, ytrain_raw, yval_raw = timeseries_ttsplit(train, train_pct=0.9)


xtrain, ytrain = (
    xtrain_raw
    .pipe(fe.split_date)
    .pipe(fe.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(fe.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(fe.join_store_details)
    .drop(["competitionopensincemonth", "competitionopensinceyear"], axis=1)
    .pipe(prep_for_model, y=ytrain_raw)  # must be last, returns x,y tuple
)

#%%
xval, yval = (
    xval_raw
    .pipe(fe.split_date)
    .pipe(fe.add_avg_customers_per_store, train_data=xtrain_raw)
    .pipe(fe.add_avg_sales_per_store, xtrain=xtrain_raw, ytrain=ytrain_raw)
    .pipe(fe.join_store_details)
    .drop(["competitionopensincemonth", "competitionopensinceyear"], axis=1)
    .pipe(prep_for_model, y=yval_raw)  # must be last, returns x,y tuple
)


meanperstore = ytrain.groupby(xtrain.store).mean().to_dict()
preds = xval.store.map(meanperstore)

print("Validation Set RMSPE")
print(rmspcte(yval, preds))

#%%
# TEST SUBMISSION ONE MEAN
if True:
    test_raw = pd.read_csv(root_path + "data/test.csv").drop('Id', axis=1).reset_index(drop=True)
    test = fix_df(test_raw.copy())

    xall_raw = pd.concat((xtrain_raw, xval_raw), axis=0)
    yall_raw = pd.concat((ytrain_raw, yval_raw), axis=0)

    xtest = (
        test
        .pipe(fe.split_date)
        .pipe(fe.add_avg_customers_per_store, train_data=xall_raw)
        .pipe(fe.add_avg_sales_per_store, xtrain=xall_raw, ytrain=yall_raw)
        .pipe(fe.join_store_details)
        .drop(['date', 'open'], axis=1)
        .rename({'customers': 'avg_store_customers'}, axis=1)
        [xval.columns]
    )

    embedding_fts = "store dayofweek dayofyear stateholiday monthofyear dayofmonth storetype assortment promointerval weekofyear".split()

    for col in embedding_fts:
        xtest[col] = xtest[col].astype('category')
        xtest[col] = xtest[col].cat.codes

    print(xtest.info())

    meanperstore = train.groupby('store')['sales'].mean().to_dict()
    preds = xtest.store.map(meanperstore)

    submission = pd.DataFrame({'Id': range(1, len(preds)+1), 'sales': preds.values.flatten()}).set_index('Id')
    submission
    submission.to_csv(root_path + "data/submissions/meanperstore_submission.csv")


#%%
