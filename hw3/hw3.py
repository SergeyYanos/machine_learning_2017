from utils import *
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer

Target_label = 'Vote'
Feature_Set = [Target_label, 'Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Most_Important_Issue',
               'Avg_Residancy_Altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters']


def train_model(data, model):
    target = data[Target_label]
    new_data = data.ix[:, data.columns != Target_label]
    # new_data = Imputer().fit_transform(new_data, target)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, new_data, target, cv=cv)
    return scores.mean(), scores.std()

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

    neural_network = MLPClassifier()                # multi layer peceptron with stochastic gradient descent

    decision_tree = DecisionTreeClassifier()        # CART decision tree (Classification and Regression Tree)

    naive_bayes = GaussianNB()                      # Gaussian naive bayes, the likelihood is using gaussian
                                                    # (exp((x-mean)/2var)/sqrt(2pi*var^2))
    knn = KNeighborsClassifier()                    # knn with k  = 5

    print train_model(data, neural_network)
    print train_model(data, decision_tree)
    print train_model(data, naive_bayes)
    print train_model(data, knn)

    # model_1, model_2 = train_two_models(train)
    # performance_1 = apply_model(test, model_1)
    # performance_2 = apply_model(test, model_2)
    # best_model = select_best_model([model_1, model_2])
    # prediction = predict(test, best_model)
    # confusion_matrix = get_confusion_matrix(test, prediction)
    # error_rate = get_error(confusion_matrix)

if __name__ == "__main__":
    main()
