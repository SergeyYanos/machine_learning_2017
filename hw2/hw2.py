import pandas
import numpy
import numbers
import os


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


# Task no. 1: Load the Election Challenge data from the ElectionsData.csv file
csv_file_path = os.path.join(os.getcwd(), "ElectionsData.csv")
data = pandas.read_csv(csv_file_path)

# Task no. 2: Identify and set the correct type of each attribute
# TODO.

# Task no. 3: Perform the following data preparation tasks using ALL the data
# 1. Imputation

print data[data.Occupation_Satisfaction.isnull()].size

for attr in ['Occupation_Satisfaction']:
    for i, v in data[data[attr].isnull()].iterrows():
        closest_i = -1
        min_dist = 100000
        r = 1
        if data[attr].dtype != 'object':
            r = data[attr].max() - data[attr].min()
        for i1, v1 in data.iterrows():
            if i != i1:
                d = dist(v, v1, r)
                if d < min_dist:
                    closest_i, min_dist = i1, d
        v[attr] = data.iloc[closest_i][attr]

print data[data.Occupation_Satisfaction.isnull()].size
