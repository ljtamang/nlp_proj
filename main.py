import os
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import csv
from decimal import Decimal


def root_dir_path():

    dir = os.path.dirname(__file__)
    return dir



def load_data(sourceFile):
    # open the file to write data
    # destinationFile = open(destinationFile, 'wb')

    with open('combined_file.csv', 'w') as outcsv:
        with open(sourceFile, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')

            writer = csv.writer(outcsv)
            headers = csvfile.readline()
            print(headers)
            writer.writerow(headers)

            for row in spamreader:
                row[-1] = str(round(float(row[-1])))
                writer.writerow(row)

def read_data(sourceFile):
    df = pd.read_csv(sourceFile)
    df['int_Gold'] = df['Gold'].round()
    print(df)


def main():

    sourceFile = root_dir_path() + "/data/train_selected-features-file-sts-2017.csv"
    destinationFile = "selected-features-file-sts-2017-training-discrete.csv"

    read_data(sourceFile)


main()
