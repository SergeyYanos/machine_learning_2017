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


def create_models():
    models = dict()

    models['Neural network'] = MLPClassifier(max_iter=2000) # multi layer peceptron with optimize gradient descent

    models['Decision tree'] = DecisionTreeClassifier()      # CART decision tree (Classification and Regression Tree)

    models['Naive bayes'] = GaussianNB()                    # Gaussian naive bayes, the likelihood is using gaussian
                                                            # (exp((x-mean)/2var)/sqrt(2pi*var^2))

    models['KNN'] = KNeighborsClassifier()                  # knn with k  = 5

    return models


def get_model_score(data, model):
    target = data[Target_label]
    new_data = data.ix[:, data.columns != Target_label]
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, new_data, target, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def select_model(data, models):
    best_score = 0
    name = str()
    for clf_name, classifier in models.iteritems():
        score, std = get_model_score(data, classifier)
        if score > best_score:
            best_score = score
            name = clf_name

    print "The Winner is", name
    print "With score:", best_score

    return models[name]


def train_model(best_model, data):
    train_label = data[Target_label]
    train_data = data.ix[:, data.columns != Target_label]
    best_model.fit(train_data, train_label)


def get_confusion_matrix(data, predict):
    test_label = data[Target_label]
    return confusion_matrix(test_label, predict)


def model_predict(best_model, data):
    test_data = data.ix[:, data.columns != Target_label]
    predict = best_model.predict(test_data)
    return predict


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


def main():
    data = read_data_to_pandas("ElectionsData.csv")

    # Prepare data set
    data = data[Feature_Set]
    set_correct_types(data)
    impute_data(data, 1000)
    cleanse_data(data)
    normalize_data(data)

    # Split the data:
    train_raw, test_raw, validate_raw = numpy.split(data, [int(.7 * len(data)), int(.9 * len(data))])
    train, test, validate = numpy.split(data, [int(.7 * len(data)), int(.9 * len(data))])

    models = create_models()
    best_model = select_model(data, models)
    train_model(best_model, train)
    prediction = model_predict(best_model, test)
    result_confusion_matrix = get_confusion_matrix(test, prediction)
    accuracy, error = get_error_and_accuracy(result_confusion_matrix)

    # model_1, model_2 = train_two_models(train)
    # performance_1 = apply_model(test, model_1)
    # performance_2 = apply_model(test, model_2)
    # best_model = select_best_model([model_1, model_2])
    # prediction = predict(test, best_model)
    # result_confusion_matrix = get_confusion_matrix(test, prediction)
    # error_rate = get_error(result_confusion_matrix)


if __name__ == "__main__":
    main()
