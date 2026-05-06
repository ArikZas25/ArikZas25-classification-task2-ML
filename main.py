import time
import json
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler # added

from sklearn.metrics import accuracy_score
from typing import List


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    
    # seperate the traning csv to featuers and labels
    df_trn = pd.read_csv(data_trn)
    X_trn = df_trn.iloc[:, :-1].values # added .values to get numpy array insted of pandas array
    y_trn = df_trn.iloc[:, -1].values  # added .values to get numpy array insted of pandas array

    # seperate the Validation csv to featuers and labels
    df_vld = pd.read_csv(data_vld)
    X_vld = df_vld.iloc[:, :-1].values # added .values to get numpy array insted of pandas array
    y_vld = df_vld.iloc[:, -1].values  # added .values to get numpy array insted of pandas array

    # seperate the Test csv to featuers and labels
    X_tst = df_tst.iloc[:, :-1].values # added .values to get numpy array insted of pandas array

    # Scaling the featuers
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn) # its already a numpy array with only featuers
    X_vld = scaler.transform(X_vld)
    X_tst = scaler.transform(X_tst)




    # todo: implement this function
    #  the data_tst dataframe should only be used for the final predictions your return
    
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    predictions = list()  # todo: return a list of your predictions for test instances
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
