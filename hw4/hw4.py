import operator
from utils import *
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture.gaussian_mixture import GaussianMixture


@timed
def train_generative_model(data, labels):
    model = GaussianNB()
    model.fit(data, labels)
    return model


@timed
def train_clustering_model(data):
    model = GaussianMixture(n_components=2, max_iter=1000, n_init=10)
    model.fit(data)
    return model


@timed
def get_voting_prediction(model, data):
    prediction = model_predict(model, data, 'Vote')
    accuracy, error = get_error_and_accuracy(get_confusion_matrix(data, prediction, 'Vote'))
    return prediction, accuracy, error


@timed
def get_clustering_prediction(model, data):
    test_data = data.ix[:, data.columns != 'Vote']
    prediction = model.predict(test_data)
    return prediction


@timed
def get_coalition(votes_distribution, cluster_histogram):
    coalition = {"parties": [], "votes": 0}
    coalition_cluster, coalition_cluster_size = 0, 0
    for cluster, histogram in cluster_histogram.iteritems():
        currents_cluster_size = sum(histogram.values())
        if currents_cluster_size > coalition_cluster_size:
            coalition_cluster, coalition_cluster_size = cluster, currents_cluster_size
    opposition_cluster = 1 - coalition_cluster
    for party in cluster_histogram[coalition_cluster]:
        if party not in cluster_histogram[opposition_cluster]:
            coalition['parties'].append(party)
            coalition['votes'] += votes_distribution[party]
    if coalition['votes'] < 51:
        votes_distribution_out_of_coalition = {k: v for k, v in votes_distribution.iteritems() if k in
                                               cluster_histogram[opposition_cluster]}
        party_percent_out_of_coalition = {}
        for party, votes in cluster_histogram[opposition_cluster].iteritems():
            try:
                party_percent_out_of_coalition[party] = \
                    float(votes) / votes + cluster_histogram[coalition_cluster][party]
            except KeyError:
                party_percent_out_of_coalition[party] = 100.0
        d = {}
        for x in votes_distribution_out_of_coalition:
            d[x] = votes_distribution_out_of_coalition[x] * party_percent_out_of_coalition[x]
        d = sorted(d.items(), key=operator.itemgetter(1))
        while coalition['votes'] < 51:
            p = d.pop()
            coalition['parties'].append(p[0])
            coalition['votes'] += p[1]

    return coalition


@timed
def get_leading_features(prediction, data):
    data['Vote'] = prediction
    leading_features = {}
    for party in set(prediction):
        leading_features[party] = data[data['Vote'] == party].var().sort_values()[:4]
    return leading_features


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
def identify_factor():
    training_set = read_data_to_pandas("train.csv")
    training_set_votes = training_set['Vote']
    training_set = training_set.drop('Vote', axis=1)
    test_set = read_data_to_pandas("test.csv")
    winning_party = {}
    features = test_set.drop("Vote", axis=1).keys()
    for feature in features:
        test_set_copy = test_set.copy(True)
        winning_party[feature] = {}
        values = test_set[feature].unique()
        for value in values:
            test_set_copy[feature] = value
            generative_model = train_generative_model(training_set, training_set_votes)
            voting_prediction, generative_accuracy_rate, generative_error_rate = \
                get_voting_prediction(generative_model, test_set_copy)
            votes_distribution = get_division_of_voters(voting_prediction)
            p_v = sorted(votes_distribution.items(), key=operator.itemgetter(1), reverse=True)[0]
            winning_party[feature][value] = (p_v[0], int(p_v[1]))

    # We can now see how each value of each feature effects the outcome of the elections.
    # For every time the winning party isn't "Yellows" we get a
    # factor (voters characteristic) which by manipulating we changed which party will win the elections.
    # In addition, for every win with 51% and greater we identified a group of factors that by being able to manipulate,
    # it will allow us to either strengthen the coalition you suggested, or construct a stronger coalition
    for feature in winning_party:
        d = {}
        for v, w in winning_party[feature].iteritems():
            if w not in d.values():
                d[v] = w
        logger.info("{0}: {1}".format(feature, d))


@timed
def main():
    # load prepared training set
    training_set = read_data_to_pandas("train.csv")
    training_set_votes = training_set['Vote']
    training_set = training_set.drop('Vote', axis=1)
    # train generative model
    generative_model = train_generative_model(training_set, training_set_votes)
    # train clustering model
    clustering_model = train_clustering_model(training_set)
    # load prepared test set
    test_set = read_data_to_pandas("test.csv")
    # apply models to test and check performance
    clustering_prediction = get_clustering_prediction(clustering_model, test_set)
    voting_prediction, generative_accuracy_rate, generative_error_rate = \
        get_voting_prediction(generative_model, test_set)
    votes_distribution = get_division_of_voters(voting_prediction)
    cluster_voters_distribution = get_clusters(clustering_prediction)
    cluster_histogram = get_cluster_voting_histogram(clustering_prediction, voting_prediction)
    # get steady coalition
    coalition = get_coalition(votes_distribution, cluster_histogram)
    # get leading features for each party
    leading_features = get_leading_features(voting_prediction, test_set)
    # identify the factor
    identify_factor()

    logger.info("Performance stats for the generative model: Accuracy Rate={0}, Error Rate={1}".
                format(generative_accuracy_rate, generative_error_rate))
    logger.info("Performance stats for the clustering model: BIC={0}, AIC={1}"
                .format(clustering_model.bic(test_set.drop('Vote', axis=1)),
                        clustering_model.aic(test_set.drop('Vote', axis=1))))
    logger.info("Votes distribution by party:")
    for party, votes in votes_distribution.iteritems():
        logger.info(party + ": " + str(votes))
    logger.info("Voters distribution by cluster:")
    for cluster, distribution in cluster_voters_distribution.iteritems():
        logger.info("%s: %s" % (cluster, distribution))
    logger.info("Voting histogram by cluster:")
    for cluster, histogram in cluster_histogram.iteritems():
        logger.info("%s: %s" % (cluster, histogram))
    logger.info("The coalition consists of: %s" % coalition)
    logger.info("Leading features by party:")
    for party, features in leading_features.iteritems():
        logger.info("%s: \n%s" % (party, features))


if __name__ == "__main__":
    main()
