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
    """
        Returns current directory path.
    """
    dir = os.path.dirname(__file__)
    return dir


def pcorrcoef(ground_truth, predictions):  # pearson correlation coefficient
    """
        Calculate  and returns pearson coefficient.
    """
    corr = np.corrcoef(ground_truth, predictions)
    return corr[0, 1]


def gold_to_discrete(source_path, desitnation_path):
    """
       Convert last gold column to discrete from continious, makes any negative value 0 and write to new csv file. 
    """
    df = pd.read_csv(source_path)
    df['Gold'] = df['Gold'].round()
    df[df < 0] = 0
    df.to_csv(desitnation_path, index=False)


def chi_sq_features_test(data_frame, no_of_features):
    """
        Perform chi-squared test and returns best number features set. 
    """
    array = data_frame.values  # values of data frame
    columns = data_frame.columns  # columns of data frame
    X = array[:, 0:(array.shape[1] - 1)]
    Y = array[:, (array.shape[1] - 1)]
    selector = SelectKBest(score_func=chi2, k=no_of_features)
    selector.fit(X, Y)
    is_selected = selector.get_support()
    selected_features = [columns[x] for x in selector.get_support(indices=True)]

    return selected_features, is_selected


def evaluate_model(model, data, target, cv=3):
    """
        Evaluate model using 10 cross fold validation and returns average score of 10 fold.
    """
    scores = cross_val_score(model, data, target, cv=cv, scoring=pcoef_scorer)
    avg_cv_score = np.mean(scores)

    return avg_cv_score


def build_GBR_model():
    """
       Build Gradient Boosting Model
    """
    params = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    model = ensemble.GradientBoostingRegressor(**params)

    return model


def initialize_result():
    """
        initialize result
    """
    result = {
        "col_names": [],  # name of features used
        "rows": [],  # If feature is selected, 1, else 0
        "best_features_by_GBR": [],  # best average GBR model accuracy
        "best_avg_GBR_score": 0.0,  # best average GBR model accuracy
        "test_score": 0.0,  # best model average GBR and SVC model accuracy
    }
    return result


def feature_selection(dataset_01, dataset_02):
    """
        Perform experiment with different set of features reported by chi-squared test and report result.
    """

    # Define size of cross validation
    kfold = 10  # kfold size

    # initialize features
    result = initialize_result()

    # for i in range(1, 3):
    for i in range(1,len(dataset_01.columns)):

        # Select best features using chi-squared test
        best_features = chi_sq_features_test(dataset_02, i)[0]  # selected features
        is_feature_selected = chi_sq_features_test(dataset_02, i)[1]  # True if selected, else false

        # Build Gradient Boosting Regression Model
        model = build_GBR_model()

        # Evaluate GBR model
        data = dataset_01[best_features].values[:, :]
        target = dataset_01.values[:, (dataset_01.values.shape[1] - 1)]
        avg_GBR_score = evaluate_model(model, data, target, cv=kfold)

        # Update  Best scores and Features using GBR model accuracy only
        if avg_GBR_score > result["best_avg_GBR_score"]:
            result["best_avg_GBR_score"] = avg_GBR_score
            result["best_features_by_GBR"] = best_features

        # Store row values (output when selecting i number of features)
        row_value = [i, avg_GBR_score] # add number of features and avg_GBR_score
        row_value.extend(map(lambda x: 1 if x else 0, is_feature_selected))
        result["rows"].append(row_value)

    col_names = ["no_features_selected", str(kfold) + "_avg_GBR_score"]
    col_names.extend(dataset_01.columns[:-1])  # exclude last gold column
    result["col_names"] = col_names

    return result


def write_result(output_path, result):
    """
         Writes result to csv file
     """

    df = pd.DataFrame(data=result["rows"], index=None, columns=result["col_names"])
    df.to_csv(path_or_buf=output_path, index=None)

    summary_data = np.array([["Selected Features: " + ", ".join(result["best_features_by_GBR"])],
                             ["Number of Features : " + str(len(result["best_features_by_GBR"]))],
                             ["model accuracy : " + str(result["best_avg_GBR_score"])],
                             ["test accuracy : " + str(result["test_score"])]
                            ])
    df2 = pd.DataFrame(data=summary_data, index=None, columns=None)
    with open(output_path, 'a') as f:
        df2.to_csv(f, header=False, index=None)

# pearson coefficient score function
pcoef_scorer = make_scorer(pcorrcoef, greater_is_better=True)


def main():
    """
     Main function of  the program
    """

    # Define path for data
    dataset_01_path = root_dir_path() + "/data/dataset_01.csv"  # data set 1
    output_path = root_dir_path() + "/result/output.csv"  # ouput data
    test_data_path = root_dir_path() + "/data/test_dataset.csv"  # test data set

    # Form data set 02 from data set 1
    # Convert gold value(continuous value) to  discrete value and negative values to zero.
    dataset_02_path = root_dir_path() + "/data/dataset_02.csv"  # data set 2
    gold_to_discrete(dataset_01_path, dataset_02_path)

    # load data set 01, data set 02 and test data
    data_set_01 = pd.read_csv(dataset_01_path)
    data_set_02 = pd.read_csv(dataset_02_path)
    test_data = pd.read_csv(test_data_path)

    # Perform feature selection
    result = feature_selection(data_set_01, data_set_02)

    # Evaluate model using test data
    data = test_data[result["best_features_by_GBR"]].values[:, :]
    target = test_data.values[:, (test_data.values.shape[1] - 1)]
    model = build_GBR_model()
    result["test_score"] = evaluate_model(model, data, target, cv=10)

    # write result
    write_result(output_path, result)

    print("Successfully executed.")


# invoke the main function
main()
