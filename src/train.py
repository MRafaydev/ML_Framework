import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_data = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_data = df[df.kfold == FOLD]

    # Target Columns
    y_train = train_data.target.values
    y_valid = valid_data.target.values

    # Features Columns
    train_data = train_data.drop(columns=["id", "target", "kfold"], axis=1)
    valid_data = valid_data.drop(columns=["id", "target", "kfold"], axis=1)
    valid_data = valid_data[train_data.columns]

    label_encoders = []
    for c in train_data.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_data[c].values.tolist() + valid_data[c].values.tolist())
        train_data.loc[:, c] = lbl.transform(train_data[c].values.tolist())
        valid_data.loc[:, c] = lbl.transform(valid_data[c].values.tolist())
        label_encoders.append((c, lbl))

    # Training
    model = RandomForestClassifier(n_jobs=-1, verbose=2)
    model.fit(train_data, y_train)
    pred = model.predict_proba(valid_data)[:, 1]
    print(metrics.roc_auc_score(y_valid, pred))
