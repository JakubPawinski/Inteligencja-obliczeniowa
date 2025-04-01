import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_iris_dataset():
    iris = datasets.load_iris()
    print(iris.target_names)

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['variety'] = [iris.target_names[i] for i in iris.target]

    all_inputs = df[iris.feature_names].values
    all_classes = df['variety'].values


    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
        all_inputs, all_classes, train_size=0.7, random_state=1)

    return train_inputs, test_inputs, train_classes, test_classes

def MLP_classifier(train_data, train_labels, hidden_layer_sizes):
    clf = MLPClassifier(solver='lbfgs', 
                        alpha=1e-5,
                        hidden_layer_sizes=hidden_layer_sizes, 
                        random_state=1,
                        max_iter=3000)

    clf.fit(train_data, train_labels) 
    return clf

def main():
    train_inputs, test_inputs, train_classes, test_classes = load_iris_dataset()


    hidden_layer_sizes = [(2,), (3,), (3, 3)]
    # hidden_layer_sizes = [(2,)]

    for hidden_layer_size in hidden_layer_sizes:
        clf = MLP_classifier(train_inputs, train_classes, hidden_layer_size)
        predict_for_test_inputs = clf.predict(test_inputs)
        predict_for_train_inputs = clf.predict(train_inputs)
        print(f"Train accuracy (for {hidden_layer_size}) train_inputs: ", accuracy_score(train_classes, predict_for_train_inputs))
        print(f"Train accuracy (for {hidden_layer_size}) test_inputs: ", accuracy_score(test_classes, predict_for_test_inputs))
    pass

if __name__ == "__main__":
    main()