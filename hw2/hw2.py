from sklearn import feature_selection
from PyAstronomy import pyasl
import matplotlib.pylab as plt
import traceback
import pandas
import numpy
import os
import sys
import time
import logging
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                           '%(''levelname)s : %(message)s',
                    datefmt="%H:%M:%S")
logger = logging.getLogger()

features_with_non_negative_values_only = [
    "Avg_monthly_expense_when_under_age_21",
    "AVG_lottary_expanses"
]

dist = {}


##########################################################
#################### Helper functions ####################
def timed(func):
    def func_wrapper(*args, **kwargs):
        start = time.time()
        logger.info("-" * 75)
        logger.info(func.func_name)
        func(*args, **kwargs)
        logger.info("-" * 75)
        logger.info("{0} - Total running time: {1} seconds".format(func.func_name, time.time() - start))

    return func_wrapper


def save_data_set_to_csv(name, data_set, sep=","):
    data_set.to_csv(name + ".csv", sep=sep, encoding="utf-8")


def plot_all_non_categorical(data_frame):
    feature_list = list(data_frame)
    feature_list = filter(lambda f: data_frame[f].dtype in ['int32', 'float64'], feature_list)
    for feature in feature_list:
        arr = data_frame[feature][data_frame[feature] != sys.maxint].as_matrix()
        plt.plot(arr, 'b.')
        neg_count = 0
        for i in range(len(arr)):
            if arr[i] < 0:
                plt.plot(i, arr[i], 'rp')
                neg_count += 1
        logger.info("feature = {0}, # negative = {1}".format(feature, neg_count))
        plt.show()


def is_nan(x):
    return x != x


def dist_tuples(x, y, r, d_types):
    d = 0.0
    for i in range(1, len(x)):
        if y[i] == sys.maxint or is_nan(y[i]):
            d += 1.0
        elif x[i] == y[i]:
            d += 0.0
        elif d_types[i] in ['int32', 'float64'] and all(x != sys.maxint and not is_nan(x) for x in [x[i], y[i]]):
            try:
                d += abs(round(x[i], 5) - round(y[i], 5)) / r[i]
            except:
                traceback.print_exc()
                logger.error("index: {0} | x[{0}]: {1} | y[{0}]: {2} | r[{0}]: {3} | d_types[{0}]: {4}".format(
                    i, x[i], y[i], r[i], d_types[i]
                ))
                logger.error("d_types:\n{0}".format(d_types))
                logger.error("x:\n{0}".format(x))
                logger.error("y:\n{0}".format(y))
                logger.error("r:\n{0}".format(r))
                sys.exit(-1)
        else:
            d += 1.0
    return d
##########################################################
######################## Solution ########################


@timed
def set_correct_types(data_frame):
    features = data_frame.keys().drop('Vote')
    for feature in features:
        logger.info(feature)
        logger.info("before - {0}".format(data_frame[feature].dtype))
        if data_frame[feature].dtype == 'object':
            data_frame[feature] = data_frame[feature].astype("category")
            data_frame[feature + "Int"] = data_frame[feature].cat.rename_categories(
                range(data_frame[feature].nunique())).astype(int)
            data_frame.loc[data_frame[feature].isnull(), feature + "Int"] = numpy.nan
        elif pandas.unique(data_frame[feature]).size < 1000:
            data_frame.loc[data_frame[feature].isnull(), feature] = sys.maxint
            data_frame[feature] = data_frame[feature].astype(int)
        else:
            data_frame.loc[numpy.isnan(data_frame[feature]), feature] = sys.maxint
        logger.info("after - {0}".format(data_frame[feature].dtype))


@timed
def impute_data(data_frame, sample_size):
    d_types = [None] + list(data_frame.dtypes)
    r = [None]
    for feature in list(data_frame):
        if data_frame[feature].dtype in ['int32', 'float64']:
            dft = data_frame[feature][data_frame[feature] != sys.maxint]
            r.append(dft.max() - dft.min())
        else:
            r.append(None)
    for feature in list(data_frame):
        logger.info("{0} - {1}".format(feature, data_frame[feature].dtype))
        logger.info("before: # missing values = {0}".format(data_frame[feature][data_frame[feature] == sys.maxint].size
                                                            if data_frame[feature].dtype in ['int32', 'float64'] else
                                                            data_frame[feature][data_frame[feature].isnull()].size))
        tuples = list(data_frame[(data_frame[feature] != sys.maxint)].dropna().itertuples())
        for row in data_frame[(data_frame[feature] == sys.maxint) | data_frame[feature].isnull()].itertuples():
            closest_i = -1
            min_dist = sys.maxint
            for i in numpy.random.choice(range(data_frame.shape[0]), sample_size):
                try:
                    other = tuples[i]
                except IndexError:
                    continue
                if row[0] != other[0]:
                    d = dist_tuples(row, other, r, d_types)
                    if d < min_dist:
                        closest_i, min_dist = other[0], d
            data_frame.set_value(index=row[0], col=feature, value=data_frame.iloc[closest_i][feature])
            if data_frame.iloc[closest_i][feature] == sys.maxint or is_nan(data_frame.iloc[closest_i][feature]):
                logger.debug("Bad value!")
                sys.exit(-1)
        logger.info("after: # missing values = {0}".format(data_frame[feature][data_frame[feature] == sys.maxint].size
                                                           if data_frame[feature].dtype in ['int32', 'float64'] else
                                                           data_frame[feature][data_frame[feature].isnull()].size))


@timed
def cleanse_data(data_frame):
    remove_noise(data_frame)
    remove_outliers(data_frame)


@timed
def remove_noise(data_frame):
    for feature in features_with_non_negative_values_only:
        logger.info("{0} - {1}".format(feature, data_frame[feature].dtype))
        before = data_frame[feature][data_frame[feature]].size
        logger.info("before: # rows = {0}".format(before))
        data_frame.drop(data_frame[data_frame[feature] < 0].index, inplace=True)
        logger.info("after: # rows = {0}".format(data_frame[feature][data_frame[feature]].size))
        logger.info("# removed rows = {0}".format(before - data_frame[feature][data_frame[feature]].size))


@timed
def remove_outliers(data_frame):
    for feature in data_frame.keys().drop('Vote'):
        if feature == 'Vote' or data_frame[feature].dtype not in ["int32", "float64"]:
            continue
        arr = data_frame[feature].as_matrix()
        logger.info("{0} - {1}".format(feature, data_frame[feature].dtype))
        logger.info("before: # rows = {0}".format(data_frame[feature].size))
        r = pyasl.generalizedESD(arr, 50, 0.05, fullOutput=True)
        data_frame.drop(data_frame.index[r[1]], inplace=True)
        logger.info("after: # rows = {0}".format(data_frame[feature].size))
        logger.info("# removed rows = {0}".format(r[0]))


DIST_NORMAL = ["Yearly_ExpensesK",
               "Yearly_IncomeK",
               "Avg_monthly_expense_on_pets_or_plants",
               "Avg_monthly_household_cost",
               "Avg_size_per_room",
               "Weighted_education_rank",
               "Political_interest_Total_Score",
               "Overall_happiness_score"]

DIST_UNIFORM = ["Financial_balance_score_(0-1)",
                "%Of_Household_Income",
                "Avg_government_satisfaction",
                "Avg_education_importance",
                "Avg_environmental_importance",
                "Avg_Satisfaction_with_previous_vote",
                "Avg_monthly_income_all_years",
                "%_satisfaction_financial_policy",
                "%Time_invested_in_work",
                "GenderInt",
                "Voting_TimeInt",
                "Age_groupInt",
                "Main_transportationInt",
                "Number_of_valued_Kneset_members",
                "Occupation_Satisfaction",
                "OccupationInt"]

DIST_DIFFERENT = ["Avg_monthly_expense_when_under_age_21",
                  "AVG_lottary_expanses",
                  "Phone_minutes_10_years",
                  "Garden_sqr_meter_per_person_in_residancy_area",
                  "Most_Important_IssueInt",
                  "MarriedInt",
                  "Will_vote_only_large_partyInt",
                  "Financial_agenda_mattersInt",
                  "Looking_at_poles_resultsInt",
                  "Avg_Residancy_Altitude",
                  "Num_of_kids_born_last_10_years",
                  "Last_school_grades",
                  "Number_of_differnt_parties_voted_for"]


@timed
def normalize_data(data_frame):
    for_uniform = data_frame[DIST_UNIFORM]
    for_normal = data_frame[DIST_NORMAL]
    for_anything_else = data_frame[DIST_DIFFERENT]

    # uniform distribution
    data_frame[for_uniform.keys()] = (
        preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(for_uniform))

    # normal distribution
    for e in for_normal.keys():
        mean = for_normal[e].mean()
        var = for_normal[e].var()
        data_frame[e] = for_normal[e].apply(lambda x: ((x - mean) / var))
    data_frame[for_normal.keys()] = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(
        for_normal)

    # every else distribution
    for e in for_anything_else.keys():
        data_frame[e] = for_anything_else[e].apply(lambda x: 0 if x == 0 else x / pow(10, numpy.ceil(numpy.log10(x))))


def feature_selection(data, labels):
    filter_method(data)
    return wrapper_method(data, labels)


def wrapper_method(data, labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    sfs = SFS(knn,
              k_features=15,
              forward=True,
              floating=False,
              verbose=2,
              scoring='accuracy',
              cv=6)
    features = sfs.fit(data.as_matrix(), labels)
    return data.iloc[:, features.k_feature_idx_]


def filter_method(data):
    threshold = 0.5
    _dict = {}
    covariance = data.cov()
    for key in covariance.keys():
        for elem in covariance[key].keys():
            if elem == key:
                continue
            covariance[key][elem] = covariance[key][elem] / (numpy.sqrt(data[key].var() * data[elem].var()))
            if covariance[key][elem] < -threshold or covariance[key][elem] > threshold:
                _dict[key] = None

    for elements in _dict.keys():
        data = data.drop(elements, axis=1)


@timed
def prepare_data_set(name, data_set):
    # Task no. 2:
    set_correct_types(data_set)

    # Task no. 3:
    # Imputation:
    impute_data(data_set, min(1000, data_set.shape[0]))

    # data_set Cleansing:
    cleanse_data(data_set)

    # Normalization (scaling):
    normalize_data(data_set)
    labels = data_set['Vote']
    data_set = data_set.drop("Vote", axis=1).select_dtypes(include=["int32", "float32", "int64", "float64"])

    # Feature Selection:
    data_set = feature_selection(data_set, labels)
    data_set['Vote'] = labels
    save_data_set_to_csv(name=name, data_set=data_set)


@timed
def main():
    # Task no. 1:
    csv_file_path = os.path.join(os.getcwd(), "ElectionsData.csv")
    data = pandas.read_csv(csv_file_path)

    # Split the data:
    train_raw, test_raw, validate_raw = numpy.split(data, [int(.7 * len(data)), int(.9 * len(data))])
    train, test, validate = train_raw.copy(), test_raw.copy(), validate_raw.copy()
    for data_set in [["train_raw", train_raw], ["test_raw", test_raw], ["validate_raw", validate_raw]]:
        save_data_set_to_csv(name=data_set[0], data_set=data_set[1])

    for data_set in [["train", train], ["test", test], ["validate", validate]]:
        prepare_data_set(data_set[0], data_set[1])


if __name__ == "__main__":
    main()
