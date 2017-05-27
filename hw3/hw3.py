from utils import *


Feature_Set = ['Yearly_Expenses', 'Yearly_Income', 'Overall_happiness_score', 'Most_Important_Issue', 'Avg residancy altitude', 'Will_vote_only_large_party', 'Financial_agenda_matters']


def main():
    data = read_data_to_pandas("ElectionsData.csv")

    data = data[Feature_Set]
    set_correct_types(data)

if __name__ == "__main__":
    main()
