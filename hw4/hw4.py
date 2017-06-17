from sklearn.naive_bayes import GaussianNB
from utils import *
from sklearn.cluster import KMeans
from sklearn.mixture.gaussian_mixture import GaussianMixture


@timed
def train_generative_model(data, labels):
    model = GaussianNB()
    model.fit(data, labels)
    return model


@timed
def train_clustering_model(data):
    model = GaussianMixture(n_components=10)
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
    return prediction  # , compute_bic(model, test_data)


@timed
def get_coalition(voting_prediction, clustering_prediction):
    raise NotImplementedError


@timed
def get_leading_features(prediction):
    raise NotImplementedError


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
    voting_prediction, generative_accuracy_rate, generative_error_rate = \
        get_voting_prediction(generative_model, test_set)
    logger.info("Performance stats for the generative model: Accuracy Rate={0}, Error Rate={1}".
                format(generative_accuracy_rate, generative_error_rate))
    logger.info("Votes distribution by party:\n{0}".format(get_division_of_voters(voting_prediction)))

    clustering_prediction = get_clustering_prediction(clustering_model, test_set)
    logger.info("Performance stats for the clustering model: BIC={0}, AIC={1}"
                .format(clustering_model.bic(test_set.drop('Vote', axis=1)),
                        clustering_model.aic(test_set.drop('Vote', axis=1))))
    logger.info("Voters distribution by cluster:\n{0}".format(get_clusters(clustering_prediction)))
    logger.info("Voting histogram by cluster:\n{0}".format(get_cluster_voting_histogram(clustering_prediction,
                                                                                        voting_prediction)))
    # get steady coalition
    # coalition = get_coalition(voting_prediction, clustering_prediction)
    # How to select coalition:
    # 1. compute dunn index for any two clusters
    # 2. number_of_votes = 0
    # 3. while number_of_votes < 51%:
    #   3.1
    # get leading features for each party
    # leading_features = get_leading_features(voting_prediction)
    # identify the factor
    # TODO
    # identify a group of factors
    # TODO


if __name__ == "__main__":
    main()
