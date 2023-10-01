import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from MCW import MCW_Algorithm
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

target_col = 'Bugged'
for j in range(1,51):
    # Preprocess source file
    metrics = pd.read_csv(f'input\\MCW{j}.csv').sort_values(
        by=['Project', 'Version'])
    metrics.replace({target_col: {True: 1, False: -1}}, inplace=True)

    results = []

    for i in metrics[['Project', 'Version']].drop_duplicates().reset_index().index:
        temp_res = []
        MCW = MCW_Algorithm()
        row = metrics[['Project', 'Version']].drop_duplicates().reset_index().iloc[i, :]
        project = row['Project']
        version = row['Version']
        # Randomly select source and target projects
        src_x, src_y, target_x, target_y = MCW.select_src_target(df=metrics, keys=['Project', 'Version'],
                                                                 target_col=target_col,
                                                                 random_state=1, ind=i)
        if src_y.value_counts().iloc[1] ==1 or src_y.value_counts().iloc[0] == 1:
            continue
        # Oversampling
        sm = SMOTE(random_state=42, k_neighbors=1)
        X_train, y_train = sm.fit_resample(src_x, src_y)

        # scale the data
        src_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=X_train.columns)
        target_scaled = pd.DataFrame(MinMaxScaler().fit_transform(target_x), columns=target_x.columns)
        target_scaled[target_col] = np.array(target_y)

        MCW.fit(src_scaled, y_train, target_scaled, target_col, random_state=1)

        X_test = target_scaled.drop(columns=[target_col])

        preds = MCW.predict(X_test)
        fpr, tpr, thresholds = roc_curve(target_y, preds)

        temp_res = [project, version, f1_score(target_y, preds), precision_score(target_y, preds),
                    recall_score(target_y, preds), accuracy_score(target_y, preds)]
        results.append(temp_res)

    res = pd.DataFrame(results, columns=['project', 'version', 'f1_score', 'precision', 'recall', 'accuracy'])
    res.to_csv(f"results_all\\MCW_res{j}.csv")
