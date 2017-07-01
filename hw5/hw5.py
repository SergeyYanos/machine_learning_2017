from utils import *


@timed
def prepare_data_set(data_set):
    set_correct_types(data_set)
    impute_data(data_set, 1000)
    cleanse_data(data_set)
    normalize_data(data_set)


@timed
def main():
    pass
    #  load the data
    train_set = read_data_to_pandas("ElectionsData.csv")
    test_set = read_data_to_pandas("ElectionsData_Pred_Features.csv")
    #  select only relevant features
    train_set = train_set[['Vote'] + Feature_Set]
    test_set = test_set[['IdentityCard_Num'] + Feature_Set]
    #  prepare data set
    prepare_data_set(train_set)
    prepare_data_set(test_set)
    train_set[Target_label].astype("category")
    #  train models(one for predicting labels and another for clustering)
    #  train classification model
    models = create_models()
    classification_model = select_model(train_set, models)
    train_model(classification_model, train_set)
    #  predict label for each voter
    voting_prediction = model_predict(classification_model, test_set)
    #  train clustering model
    cluster_model = train_clustering_model(train_set.drop(['Vote'], axis=1))
    #  predict cluster for each voter
    clustering_prediction = get_clustering_prediction(cluster_model, test_set.drop(['IdentityCard_Num'], axis=1))
    cluster_histogram = get_cluster_voting_histogram(clustering_prediction, voting_prediction)
    cluster_voters_distribution = get_clusters(clustering_prediction)
    #  get division of votes
    votes_distribution = get_division_of_voters(voting_prediction)
    #  get winning party
    winning_party = get_winning_party(votes_distribution)
    #  get steady coalition
    coalition = get_coalition(votes_distribution, cluster_histogram)

    logger.info("Votes distribution by party:")
    for party, votes in votes_distribution.iteritems():
        logger.info(party + ": " + str(votes))
    logger.info("Winning party: " + str(winning_party))
    logger.info("Voters distribution by cluster:")
    for cluster, distribution in cluster_voters_distribution.iteritems():
        logger.info("%s: %s" % (cluster, distribution))
    logger.info("Voting histogram by cluster:")
    for cluster, histogram in cluster_histogram.iteritems():
        logger.info("%s: %s" % (cluster, histogram))
    logger.info("The coalition consists of: %s" % coalition)

if __name__ == "__main__":
    main()
