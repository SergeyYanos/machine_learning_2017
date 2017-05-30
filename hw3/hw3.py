import csv
import operator
from collections import Counter

from utils import *
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

Target_label = 'Vote'
Feature_Set = [Target_label, 'Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Most_Important_Issue',
               'Avg_Residancy_Altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters']


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
    test_data = data.ix[:, data.columns != Target_label]
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


def get_most_probable_transport(prediction, used_transport):
    most_probable_transport = {}
    transports_used = {}
    for i in range(len(prediction)):
        if prediction[i] not in transports_used:
            transports_used[prediction[i]] = []
        transports_used[prediction[i]].append(used_transport[i])
    for party, transport_used in transports_used.iteritems():
        counter = Counter(transport_used).most_common(1)[0]
        most_probable_transport[party] = (counter[0], round(100 * float(counter[1]) / len(transport_used), 2))
    return most_probable_transport


@timed
def main():
    data = read_data_to_pandas("ElectionsData.csv")
    data_raw = data[Feature_Set]
    data_prepared = data[Feature_Set + ['Main_transportation']]

    # Prepare data set
    set_correct_types(data_prepared)
    data_prepared['Main_transportation'].astype("category")
    impute_data(data_prepared, 1000)
    cleanse_data(data_prepared)
    normalize_data(data_prepared)
    data_prepared[Target_label].astype("category")

    # Split the data:
    train_raw, test_raw, validate_raw = numpy.split(data_raw, [int(.7 * len(data_raw)), int(.9 * len(data_raw))])
    train, test, validate = numpy.split(data_prepared, [int(.7 * len(data_prepared)), int(.9 * len(data_prepared))])

    test_transport = test[['Main_transportation']]
    test = test[Feature_Set]
    train = train[Feature_Set]
    validate = validate[Feature_Set]

    # Save to csv
    for data_set in [["train_raw", train_raw], ["test_raw", test_raw], ["validate_raw", validate_raw],
                     ["train", train], ["test", test], ["validate", validate]]:
        save_data_set_to_csv(name=data_set[0], data_set=data_set[1])

    # Create 4 models
    models = create_models()
    # Select the best model
    best_model = select_model(train, models)
    train_model(best_model, train)
    # Predict via the best model
    prediction = model_predict(best_model, test)
    # Create confusion matrix
    result_confusion_matrix = get_confusion_matrix(test, prediction)
    # Calculate accuracy and error rates
    accuracy, error = get_error_and_accuracy(result_confusion_matrix)
    # Calculate division of voters
    division_of_voters = get_division_of_voters(prediction)
    # Get the winning party
    winners = get_winning_party(division_of_voters)
    # Get most probable transport
    most_probable_transport = get_most_probable_transport(prediction, test_transport['Main_transportation'].tolist())

    logger.info("Our models confusion matrix:\n{0}".format(result_confusion_matrix))
    logger.info("Our model has {0} accuracy rate and {1} error rate".format(accuracy, error))
    logger.info("The winning party is {party} with {percent}% of votes".format(party=winners[0], percent=winners[1]))
    logger.info("The division of votes is: {0}".format(division_of_voters))
    logger.info("Most probable means of transport per party:\n{0}".format(most_probable_transport))

    # Save predictions to csv
    with open("predictions.csv", 'wb') as predictions:
        wr = csv.writer(predictions, quoting=csv.QUOTE_ALL)
        wr.writerow(prediction)


if __name__ == "__main__":
    main()
