import pandas
import numpy
import numbers
import os
import sys


def dist(x, y, r):
    d = 0.0
    for attr in x.keys():
        if x[attr] == y[attr]:
            d += 0.0
        elif isinstance(x[attr], numbers.Number) and not numpy.isnan(x[attr]) and not numpy.isnan(y[attr]):
            d += float(abs(x[attr] - y[attr]) / r)
        else:
            d += 1.0
    return d


def set_correct_types(data_frame):
    obj_features = data_frame.keys()[data_frame.dtypes.map(lambda x: x == 'object')].drop('Vote')
    for f in obj_features:
        data_frame[f] = data_frame[f].astype("category")
        data_frame[f + "Int"] = data_frame[f].cat.rename_categories(range(data_frame[f].nunique())).astype(int)
        data_frame.loc[data_frame[f].isnull(), f + "Int"] = numpy.nan

    not_obj_features = data_frame.keys()[data_frame.dtypes.map(lambda x: x != 'object')]
    for f in not_obj_features:
        if pandas.unique(data_frame[f]).size < 1000:
            data_frame[f] = data_frame[f].astype("category")
            data_frame[f] = data_frame[f].cat.rename_categories(range(data_frame[f].nunique())).astype(int)
            data_frame.loc[data_frame[f].isnull(), f] = sys.maxint


def impute_data(data_frame):
    print data_frame[pandas.isnull(data_frame).any(axis=1)]
    # for i, v in data_frame[data_frame.isnull()].iterrows():
    #     closest_i = -1
    #     min_dist = sys.maxint
    #     r = 1
    #     if data[attr].dtype != 'object':
    #         r = data[attr].max() - data[attr].min()
    #     for i1, v1 in data.iterrows():
    #         if i != i1:
    #             d = dist(v, v1, r)
    #             if d < min_dist:
    #                 closest_i, min_dist = i1, d
    #     v[attr] = data.iloc[closest_i][attr]

if __name__ == "__main__":
    # Task no. 1: Load the Election Challenge data from the ElectionsData.csv file
    csv_file_path = os.path.join(os.getcwd(), "ElectionsData.csv")
    data = pandas.read_csv(csv_file_path)

    # Task no. 2: Identify and set the correct type of each attribute.
    set_correct_types(data)

    # Task no. 3: Perform the following data preparation tasks using ALL the data
    # Imputation:
    impute_data(data)