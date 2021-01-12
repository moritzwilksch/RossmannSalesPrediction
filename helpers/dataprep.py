#%%
from catboost import CatboostError
import pandas as pd
root_path = "../"

def fix_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to lower case.
    Fixes `object` data types
    """
    data = data.rename({col: col.lower() for col in data.columns}, axis=1)
    data["date"] = data['date'].astype('datetime64')
    data['stateholiday'] = (
        data['stateholiday']
        .map({0: 'None', '0': 'no', 'a': 'public', 'b': 'easter', 'c': 'xmas'})
        .astype('category')
        )


    return data






#%%
if __name__ == '__main__':
    import numpy as np
    def rmspcte(real, pred):
        mask = ~(real == 0)
        real, pred = real[mask], pred[mask]
        return np.sqrt(np.mean(((real.values-pred.values)/real.values)**2))**0.5

    train = pd.read_csv(root_path + 'data/train.csv')
    train = fix_df(train)
    print(train.stateholiday.unique())

    test = pd.read_csv(root_path + 'data/test.csv')
    test = fix_df(test)

    #train_test_thresh = train.date.min() + pd.Timedelta(753, unit="d")
    #test = train.query("date >= @train_test_thresh")
    #train = train.query("date < @train_test_thresh")


    model = train.groupby('store')['sales'].mean().to_dict()

    preds = test.store.map(model)

    preds.index = preds.index+1
    print(preds)
    preds.to_csv(root_path + 'submission_meanonly.csv')
    




