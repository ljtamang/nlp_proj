import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics.scorer import make_scorer
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm

import csv
from decimal import Decimal


def root_dir_path():
    dir = os.path.dirname(__file__)
    return dir


def pcorrcoef(ground_truth, predictions):  # pearson correlation coefficient
    corr = np.corrcoef(ground_truth, predictions)
    return corr[0, 1]


def gold_to_discrete(source_path, desitnation_path):
    df = pd.read_csv(source_path)
    df['Gold'] = df['Gold'].round()
    df[df < 0] = 0
    df.to_csv(desitnation_path, index=False)


def chi_sq_features_test(data_frame, no_of_features):
    array = data_frame.values  # values of data frame
    columns = data_frame.columns  # columns of data frame
    X = array[:, 0:(array.shape[1] - 1)]
    Y = array[:, (array.shape[1] - 1)]
    selector = SelectKBest(score_func=chi2, k=no_of_features)
    selector.fit(X, Y)
    is_selected = selector.get_support()
    selected_features = [columns[x] for x in selector.get_support(indices=True)]

    return selected_features, is_selected

pcoef_scorer = make_scorer(pcorrcoef, greater_is_better=True)


def main():

    # Define size of cross validation
    kfold = 10  # kfold size

    # Path of data set 01
    output_path = root_dir_path() + "/result/a.csv"

    # Path of data set 01
    dataset_01_path = root_dir_path() + "/data/selected-features-file-sts-2017-training.csv"

    # Form data set 02
    # Convert gold value(continuous value) to  discrete value and negative values to zero.
    dataset_02_path = root_dir_path() + "/data/dataset_02.csv"
    gold_to_discrete(dataset_01_path, dataset_02_path)

    # load dataset 01 and dataset 02
    dataset_01 =  pd.read_csv(dataset_01_path)
    dataset_02 = pd.read_csv(dataset_02_path)

    # Best Features Selection
    # Each CMi and RMi model will be evaluated using 10 - fold cross validation.
    # The feature set Fi will be selected as best feature if average accuracy of CMi and RMI is best.
    outputs = dict()
    best_avg_accuracy = 0.0

    values = []  # collection of row values

    # for i in range(1,len(dataset_01.columns)):
    for i in range(1, 3):

        # find best i number of features
        Fi = chi_sq_features_test(dataset_02,i)[0]  # selected features
        is_selected = chi_sq_features_test(dataset_02, i)[1]  # True if selected, else false

        # build Gradient Boosting Regression Model, RMi
        params = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)
        GBRMi = ensemble.GradientBoostingRegressor(**params)

        # build Support Vector Classification model, CMi
        # CMi = svm.SVC(kernel='linear')

        # Evaluate GBR model
        data = dataset_01[Fi].values[:, :]
        target = dataset_01.values[:, (dataset_01.values.shape[1] - 1)]
        scores = cross_val_score(GBRMi, data, target, cv=kfold, scoring=pcoef_scorer)
        folds_avg_accuracy_r = np.mean(scores)
        # print(folds_avg_accuracy_r)

        '''
        # Evaluate classification model
        target = dataset_02.values[:, (dataset_02.values.shape[1] - 1)]  # gold values
        scores = cross_val_score(CMi, data, target, cv=kfold, scoring=pcoef_scorer)
        folds_avg_accuracy_c = np.mean(scores)

        # Average accuracy of regression and classification
        folds_avg_accuracy = np.mean([folds_avg_accuracy_r,folds_avg_accuracy_c ])
        '''

        # update best accuracy and features
        if folds_avg_accuracy_r> best_avg_accuracy:
            best_avg_accuracy = folds_avg_accuracy_r
            best_feature = Fi

        # store row values
        row_value = [i, folds_avg_accuracy_r]
        row_value.extend(map(lambda x: 1 if x else 0, is_selected)) # uncomment if you want to add 0 and 1
        # row_value.extend(is_selected)
        values.append(row_value)
        print(row_value)
        print(len(row_value))

    col_names = ["no_features_selected", str(kfold)+"_fold_avg_accuracy"]
    col_names.extend(dataset_01.columns[:-1]) # exclude last gold column
    df = pd.DataFrame(data=values, index=None, columns=col_names)
    df.to_csv(path_or_buf=output_path, index=None)


main()
