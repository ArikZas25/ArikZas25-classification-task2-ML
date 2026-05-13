import time
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    # Data Loading and Scaling
    df_trn = pd.read_csv(data_trn)
    X_trn = df_trn.iloc[:, :-1].values
    y_trn = df_trn.iloc[:, -1].values

    df_vld = pd.read_csv(data_vld)
    X_vld = df_vld.iloc[:, :-1].values
    y_vld = df_vld.iloc[:, -1].values

    X_tst = df_tst.values

    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_vld = scaler.transform(X_vld)
    X_tst = scaler.transform(X_tst)

    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # Calculate all Euclidean distances from validation instances to training instances
    val_dists = np.array([np.linalg.norm(X_trn - x, axis=1) for x in X_vld])

    # Generate a list of potential r to test dynamically based on distance distribution
    min_dist = np.min(val_dists)
    max_dist = np.percentile(val_dists, 20)  # Test up to the 20th percentile
    candidate_radii = np.linspace(min_dist + 0.01, max_dist, num=20)

    best_radius = candidate_radii[0]
    best_accuracy = -1.0

    # find the best radius
    for r in candidate_radii:
        y_pred_vld = []
        for i in range(len(X_vld)):
            # Find indices of training instances within the current radius
            neighbors_idx = np.where(val_dists[i] <= r)[0]

            if len(neighbors_idx) > 0:
                # apply majority vote
                neighbors_labels = y_trn[neighbors_idx]
                vals, counts = np.unique(neighbors_labels, return_counts=True)
                predicted_label = vals[np.argmax(counts)]
            else:
                # fallback to 1-Nearest Neighbor
                closest_idx = np.argmin(val_dists[i])
                predicted_label = y_trn[closest_idx]

            y_pred_vld.append(predicted_label)

        current_acc = accuracy_score(y_vld, y_pred_vld)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_radius = r

    print(f'Optimal radius selected: {best_radius:.4f}')

    # Generate predictions for the Test set

    predictions = []

    for i in range(len(X_tst)):
        # Calculate distances from the current test instance to all training instances
        dists = np.linalg.norm(X_trn - X_tst[i], axis=1)
        neighbors_idx = np.where(dists <= best_radius)[0]

        if len(neighbors_idx) > 0:
            # apply majority vote
            neighbors_labels = y_trn[neighbors_idx]
            vals, counts = np.unique(neighbors_labels, return_counts=True)
            predicted_label = vals[np.argmax(counts)]
        else:
            # fallback to Nearest Neighbor
            closest_idx = np.argmin(dists)
            predicted_label = y_trn[closest_idx]

        predictions.append(predicted_label)

    return predictions

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty predictions, should not happen
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'classification accuracy on the test set: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
