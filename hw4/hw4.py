from utils import *


@timed
def train_generative_model(data):
    raise NotImplementedError


@timed
def train_clustering_model(data):
    raise NotImplementedError


@timed
def get_voting_prediction(model, data):
    raise NotImplementedError


@timed
def get_clustering_prediction(model, data)
    raise NotImplementedError

@timed
def get_coalition(voting_prediction, clustering_prediction):
    raise NotImplementedError

@timed
def get_leading_features(prediction):
    raise NotImplementedError


@timed
def main():
    # load prepared training set
    training_set = read_data_to_pandas("train.csv")
    # train generative model
    generative_model = train_generative_model(training_set)
    # train clustering model
    clustering_model = train_clustering_model(training_set)
    # load prepared test set
    test_set = read_data_to_pandas("test.csv")
    # apply models to test and check performance
    voting_prediction = get_voting_prediction(generative_model, test_set)
    clustering_prediction = get_clustering_prediction(clustering_model, test_set)
    # get steady coalition
    coalition = get_coalition(voting_prediction, clustering_prediction)
    # get leading features for each party
    leading_features = get_leading_features(voting_prediction)
    # identify the factor
    # TODO
    # identify a group of factors
    # TODO


if __name__ == "__main__":
    main()
