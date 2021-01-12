#%%
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").resolve().parent.parent))

from rich.console import Console
c = Console(highlight=False)
import numpy as np
import pandas as pd
from RossmannSalesPrediction.helpers.evaluation import rmspcte
from RossmannSalesPrediction.helpers.dataprep import fix_df

root_path = "../"

#%%
train = pd.read_csv(root_path + 'data/train.csv')
train = fix_df(train)
c.print(train.stateholiday.unique())

test = pd.read_csv(root_path + 'data/test.csv')
test = fix_df(test)



model = train.groupby('store')['sales'].mean().to_dict()
preds = test.store.map(model)

preds.index = preds.index+1
c.print(preds)

#%%
preds.to_csv(root_path + 'submission_meanonly.csv')