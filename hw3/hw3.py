from utils import *

Feature_Set = ['Vote', 'Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Most_Important_Issue',
               'Avg_Residancy_Altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters']


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

    # model_1, model_2 = train_two_models(train)
    # performance_1 = apply_model(test, model_1)
    # performance_2 = apply_model(test, model_2)
    # best_model = select_best_model([model_1, model_2])
    # prediction = predict(test, best_model)
    # confusion_matrix = get_confusion_matrix(test, prediction)
    # error_rate = get_error(confusion_matrix)

if __name__ == "__main__":
    main()
