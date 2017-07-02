from mlxtend.evaluate import confusion_matrix
from sklearn import feature_selection
from PyAstronomy import pyasl
import matplotlib.pylab as plt
import traceback
import pandas
import numpy
import sys
import os
import time
import logging
import operator
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-5s %(name)-5s %(threadName)-5s %(filename)s:%(lineno)s - %(funcName)s() '
                           '%(''levelname)s : %(message)s',
                    datefmt="%H:%M:%S")
logger = logging.getLogger()

features_with_non_negative_values_only = [
    "Avg_monthly_expense_when_under_age_21",
    "AVG_lottary_expanses"
]

Target_label = 'Vote'
Feature_Set = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Most_Important_Issue',
               'Avg_Residancy_Altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters']


def read_data_to_pandas(path):
    csv_file_path = os.path.join(os.getcwd(), path)
    data = pandas.read_csv(csv_file_path)
    return data


def timed(func):
    def func_wrapper(*args, **kwargs):
        start = time.time()
        logger.info("-" * 75)
        logger.info(func.func_name)
        rc = func(*args, **kwargs)
        logger.info("-" * 75)
        logger.info("{0} - Total running time: {1} seconds".format(func.func_name, time.time() - start))
        return rc

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


@timed
def set_correct_types(data_frame):
    if 'Vote' in data_frame.keys():
        features = data_frame.keys().drop(['Vote'])
    else:
        features = data_frame.keys()
    for feature in features:
        logger.info(feature)
        logger.info("before - {0}".format(data_frame[feature].dtype))
        if data_frame[feature].dtype == 'object':
            data_frame[feature] = data_frame[feature].astype("category")
            data_frame[feature] = data_frame[feature].cat.rename_categories(
                range(data_frame[feature].nunique())).astype(int)
            data_frame.loc[data_frame[feature].isnull(), feature] = numpy.nan
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
        if feature in data_frame:
            logger.info("{0} - {1}".format(feature, data_frame[feature].dtype))
            before = data_frame[feature][data_frame[feature]].size
            logger.info("before: # rows = {0}".format(before))
            data_frame.drop(data_frame[data_frame[feature] < 0].index, inplace=True)
            logger.info("after: # rows = {0}".format(data_frame[feature][data_frame[feature]].size))
            logger.info("# removed rows = {0}".format(before - data_frame[feature][data_frame[feature]].size))


@timed
def remove_outliers(data_frame):
    if 'Vote' in data_frame.keys():
        features = data_frame.keys().drop(['Vote'])
    else:
        features = data_frame.keys()
    for feature in features:
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
                "Gender",
                "Voting_Time",
                "Age_group",
                "Number_of_valued_Kneset_members",
                "Occupation_Satisfaction",
                "Occupation"]

DIST_DIFFERENT = ["Avg_monthly_expense_when_under_age_21",
                  "AVG_lottary_expanses",
                  "Phone_minutes_10_years",
                  "Garden_sqr_meter_per_person_in_residancy_area",
                  "Most_Important_Issue",
                  "Married",
                  "Will_vote_only_large_party",
                  "Financial_agenda_matters",
                  "Looking_at_poles_results",
                  "Avg_Residancy_Altitude",
                  "Num_of_kids_born_last_10_years",
                  "Last_school_grades",
                  "Number_of_differnt_parties_voted_for"]


@timed
def normalize_data(data_frame):
    for_uniform = data_frame[filter(lambda x: x in data_frame, DIST_UNIFORM)]
    for_normal = data_frame[filter(lambda x: x in data_frame, DIST_NORMAL)]
    for_anything_else = data_frame[filter(lambda x: x in data_frame, DIST_DIFFERENT)]

    # uniform distribution
    if len(for_uniform.keys()):
        data_frame[for_uniform.keys()] = (
            preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(for_uniform))

    # normal distribution
    if len(for_normal.keys()):
        for e in for_normal.keys():
            mean = for_normal[e].mean()
            var = for_normal[e].var()
            data_frame[e] = for_normal[e].apply(lambda x: 0 if var == 0 else ((x - mean) / var))
        data_frame[for_normal.keys()] = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(
            for_normal)

    # every else distribution
    if len(for_anything_else.keys()):
        for e in for_anything_else.keys():
            data_frame[e] = for_anything_else[e].apply(lambda x: 0 if x == 0 else x / pow(10, numpy.ceil(numpy.log10(x))))
        data_frame[for_anything_else.keys()] = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(
            for_anything_else)


@timed
def prepare_data_set(data_set):
    set_correct_types(data_set)
    impute_data(data_set, 1000)
    cleanse_data(data_set)
    normalize_data(data_set)


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
            if data[key].var() == 0 or data[elem].var() == 0:
                covariance[key][elem] = 0
            else:
                covariance[key][elem] = covariance[key][elem] / (numpy.sqrt(data[key].var() * data[elem].var()))

            if covariance[key][elem] < -threshold or covariance[key][elem] > threshold:
                _dict[key] = None

    for elements in _dict.keys():
        data = data.drop(elements, axis=1)

@timed
def create_models():
    models = dict()

    # multi layer perceptron with optimize gradient descent
    models['Neural network'] = MLPClassifier(max_iter=2000)
    # CART decision tree (Classification and Regression Tree)
    models['Decision tree'] = DecisionTreeClassifier()
    # Gaussian naive bayes, the likelihood is using gaussian
    # (exp((x-mean)/2var)/sqrt(2pi*var^2))
    models['Naive bayes'] = GaussianNB()
    # knn with k  = 5
    models['KNN'] = KNeighborsClassifier()

    return models


@timed
def get_model_score(data, model):
    target = data[Target_label]
    new_data = data.ix[:, data.columns != Target_label]
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, new_data, target, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


@timed
def select_model(data, models):
    best_score = 0
    name = str()
    for clf_name, classifier in models.iteritems():
        score, std = get_model_score(data, classifier)
        if score > best_score:
            best_score = score
            name = clf_name

    logger.info("The best model is {name} with score {score}".format(name=name, score=best_score))

    return models[name]


@timed
def train_model(best_model, data):
    train_label = data[Target_label]
    train_data = data.ix[:, data.columns != Target_label]
    best_model.fit(train_data, train_label)


@timed
def get_confusion_matrix(data, predict):
    test_label = data[Target_label]
    return confusion_matrix(test_label, predict)


@timed
def model_predict(best_model, data):
    test_data = data.ix[:, data.columns != 'IdentityCard_Num']
    predict = best_model.predict(test_data)
    return predict


@timed
def get_error_and_accuracy(data_confusion_matrix):
    accuracy = 0
    error = 0
    for i in range(data_confusion_matrix.shape[0]):
        for j in range(data_confusion_matrix.shape[1]):
            if i == j:
                accuracy += data_confusion_matrix[i][j]
            else:
                error += data_confusion_matrix[i][j]
    total = accuracy + error
    return float(accuracy) / total, float(error) / total


@timed
def get_winning_party(division_of_votes):
    return sorted(division_of_votes.items(), key=operator.itemgetter(1), reverse=True)[0]


@timed
def get_division_of_voters(prediction):
    division_of_votes = {}
    for vote in prediction:
        if vote not in division_of_votes:
            division_of_votes[vote] = 0
        division_of_votes[vote] += 1
    total_number_of_votes = len(prediction)
    for party, votes in division_of_votes.iteritems():
        division_of_votes[party] = round((float(votes) / total_number_of_votes) * 100, 2)
    return division_of_votes


@timed
def train_clustering_model(data):
    model = GaussianMixture(n_components=2, max_iter=1000, n_init=10)
    model.fit(data)
    return model


@timed
def get_clustering_prediction(model, data):
    test_data = data.ix[:, data.columns != 'Vote']
    prediction = model.predict(test_data)
    return prediction


@timed
def get_coalition(votes_distribution, cluster_histogram):
    coalition = {"parties": [], "votes": 0.0}
    coalition_cluster, coalition_cluster_size = 0, 0
    for cluster, histogram in cluster_histogram.iteritems():
        currents_cluster_size = sum(histogram.values())
        if currents_cluster_size > coalition_cluster_size:
            coalition_cluster, coalition_cluster_size = cluster, currents_cluster_size
    opposition_cluster = 1 - coalition_cluster
    sorted_votes_distribution = sorted(votes_distribution.items(), reverse=True, key=operator.itemgetter(1))
    for party in sorted_votes_distribution:
        if party[0] not in cluster_histogram[opposition_cluster]:
            coalition['parties'].append(party[0])
            coalition['votes'] += party[1]
            if coalition['votes'] >= 51:
                break

    if coalition['votes'] < 51:
        logger.info(coalition)
        logger.info("Not enough votes in the base coalition, adding more parties")
        votes_distribution_out_of_coalition = {k: v for k, v in votes_distribution.iteritems() if k in
                                               cluster_histogram[opposition_cluster]}
        party_percent_out_of_coalition = {}
        for party, votes in cluster_histogram[opposition_cluster].iteritems():
            try:
                party_percent_out_of_coalition[party] = \
                    float(votes) / (votes + cluster_histogram[coalition_cluster][party])
            except KeyError:
                party_percent_out_of_coalition[party] = 100.0
        d = {}
        for x in votes_distribution_out_of_coalition:
            d[x] = votes_distribution_out_of_coalition[x] * party_percent_out_of_coalition[x]
            d[x] /= 100
        d = sorted(d.items(), reverse=True, key=operator.itemgetter(1))
        logger.info(d)
        while coalition['votes'] < 51:
            p = d.pop()
            coalition['parties'].append(p[0])
            coalition['votes'] += votes_distribution[p[0]]
            logger.info(coalition)

    return coalition


@timed
def get_cluster_voting_histogram(clusters, votes):
    histogram = {}
    for i in range(len(clusters)):
        if clusters[i] not in histogram:
            histogram[clusters[i]] = {}
        if votes[i] not in histogram[clusters[i]]:
            histogram[clusters[i]][votes[i]] = 0
        histogram[clusters[i]][votes[i]] += 1
    return histogram


@timed
def get_clusters(prediction):
    clusters = {}
    total = len(prediction)
    for x in prediction:
        if x not in clusters:
            clusters[x] = 0
        clusters[x] += 1
    for k, v in clusters.iteritems():
        clusters[k] = round((float(v) / total) * 100, 2)
    return clusters
