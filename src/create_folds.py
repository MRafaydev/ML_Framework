from unicodedata import name
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # loading dataset
    df = pd.read_csv('../input/train.csv')
    # Initializing the new dataframe 
    df['kfold'] = -1
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # splitting the data into 10 folds
    kf = model_selection.KFold(n_splits=5, shuffle=False)
    # iterating through the folds
    for folds, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = folds
    # saving the dataframe to csv
    df.to_csv('../input/train_folds.csv', index=False)