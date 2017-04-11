import pandas
import numpy
import numbers
import os
import sys

import time


def dist(x, y, r):
    d = 0.0
    for attr in x.keys():
        if x[attr] == y[attr]:
            d += 0.0
        elif isinstance(x[attr], numbers.Number) \
                and not numpy.isnan(x[attr]) and not numpy.isnan(y[attr]) \
                and x[attr] != sys.maxint and y[attr] != sys.maxint:
            d += float(abs(x[attr] - y[attr]) / r)
        else:
            d += 1.0
    return d


def set_correct_types(data_frame):
    features = data_frame.keys().drop('Vote')
    for f in features:
        if data_frame[f].dtype == 'object':
            data_frame[f] = data_frame[f].astype("category")
            data_frame[f + "Int"] = data_frame[f].cat.rename_categories(range(data_frame[f].nunique())).astype(int)
            data_frame.loc[data_frame[f].isnull(), f + "Int"] = numpy.nan
        elif pandas.unique(data_frame[f]).size < 1000:
            data_frame.loc[data_frame[f].isnull(), f] = sys.maxint
            data_frame[f] = data_frame[f].astype(int)


def impute_data(data_frame):
    iter_rows = data_frame.iterrows()
    for attr in list(data_frame):
        for i, v in data_frame[(data_frame[attr] == sys.maxint) | (data_frame[attr].isnull())].iterrows():
            print i, attr
            print "before:", data_frame.iloc[i][attr]
            closest_i = -1
            min_dist = sys.maxint
            if data_frame[attr].dtype in ['int32', 'float64']:
                r = data_frame[attr].max() - data_frame[attr].min()
            else:
                r = None
            start = time.time()
            for i1, v1 in iter_rows:
                if i != i1:
                    d = dist(v, v1, r)
                    if d < min_dist:
                        closest_i, min_dist = i1, d
            print "time:", time.time() - start
            data_frame.set_value(index=i, col=attr, value=data_frame.iloc[closest_i][attr])
            print "after:", data_frame.iloc[i][attr]

if __name__ == "__main__":
    # Task no. 1: Load the Election Challenge data from the ElectionsData.csv file
    csv_file_path = os.path.join(os.getcwd(), "ElectionsData.csv")
    data = pandas.read_csv(csv_file_path)

    # Task no. 2: Identify and set the correct type of each attribute.
    set_correct_types(data)

    # Task no. 3: Perform the following data preparation tasks using ALL the data
    # Imputation:
    impute_data(data)
