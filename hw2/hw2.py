import pandas
import numpy
import numbers
import os
import sys
import time


def dist_tuples(x, y, r):
    d = 0.0
    for i in range(len(x)):
        if y[i] == sys.maxint:
            d += 1.0
        elif x[i] == y[i]:
            d += 0.0
        elif isinstance(x[i], numbers.Number) and x[i] != sys.maxint:
            d += abs(x[i] - y[i]) / r
        else:
            d += 1.0
    return d


def dist_rows(x, y, r):
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
            data_frame[f] = data_frame[f].cat.rename_categories(range(data_frame[f].nunique())).astype(int)
            data_frame.loc[data_frame[f].isnull(), f] = sys.maxint
        elif pandas.unique(data_frame[f]).size < 1000:
            data_frame.loc[data_frame[f].isnull(), f] = sys.maxint
            data_frame[f] = data_frame[f].astype(int)


def impute_data(data_frame):
    for attr in list(data_frame):
        for v in data_frame[(data_frame[attr] == sys.maxint)].itertuples():
            closest_i = -1
            min_dist = sys.maxint
            if data_frame[attr].dtype in ['int32', 'float64']:
                dft = data_frame[attr][data_frame[attr] != sys.maxint]
                r = dft.max() - dft.min()
            else:
                r = None
            for v1 in data_frame.itertuples():
                if v[0] != v1[0]:
                    d = dist_tuples(v, v1, r)
                    if d < min_dist:
                        closest_i, min_dist = v1[0], d
                        # print closest_i, min_dist
            data_frame.set_value(index=v[0], col=attr, value=data_frame.iloc[closest_i][attr])
            if data_frame.iloc[v[0]][attr] == sys.maxint:
                print "!!!!!!", attr, v[0], closest_i, min_dist, "!!!!!!"
                return

if __name__ == "__main__":
    # Task no. 1: Load the Election Challenge data from the ElectionsData.csv file
    csv_file_path = os.path.join(os.getcwd(), "ElectionsData.csv")
    data = pandas.read_csv(csv_file_path)

    # Task no. 2: Identify and set the correct type of each attribute.
    set_correct_types(data)

    # Task no. 3: Perform the following data preparation tasks using ALL the data
    # Imputation:
    start = time.time()
    impute_data(data)
    print "total time:", time.time() - start

    # for v in data.itertuples():
    #     if v[0] == 5491:
    #         closest_i = -1
    #         min_dist = sys.maxint
    #         for v1 in data.itertuples():
    #             if v[0] != v1[0]:
    #                 r = data['Number_of_differnt_parties_voted_for'][data['Number_of_differnt_parties_voted_for'] != sys.maxint].max() - \
    #                     data['Number_of_differnt_parties_voted_for'][data['Number_of_differnt_parties_voted_for'] != sys.maxint].min()
    #                 d = dist_tuples(v, v1, r)
    #                 if 0 <= d < min_dist:
    #                     closest_i, min_dist = v1[0], d
    #                     print closest_i, min_dist
    #                     print v1[2], v1[2] == sys.maxint
    #                     # print abs(v1[i] - v[i])
    #                     print r
