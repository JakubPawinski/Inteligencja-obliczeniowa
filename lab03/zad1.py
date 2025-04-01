import pandas as pd
from sklearn.model_selection import train_test_split


def classify_iris(sl, sw, pl, pw):
    if pl < 1.9 and pw < 0.6:
        return("Setosa")
    elif pl > 4.8:
        return("Virginica")
    else:
        return("Versicolor")



def print_prediction(train_set, test_set):
    good_predictions = 0
    len = test_set.shape[0]
    for i in range(len):
        if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, 4]:
            good_predictions = good_predictions + 1
    print(good_predictions)
    print(good_predictions/len*100, "%")

def main():
    df = pd.read_csv("iris1 1.csv")
    (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292628)

    # test(train_set)
    train_df = pd.DataFrame(train_set, columns=df.columns)

    print(train_df.sort_values(by="variety").to_string())

    print_prediction(train_set, test_set)

if __name__ == "__main__":
    main()

