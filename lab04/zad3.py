import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_diabetes_dataset():
    df = pd.read_csv('diabetes 1.csv')

    # print(df.columns)

    all_inputs = df.drop(columns=['class']).values
    all_classes = df['class'].values

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
        all_inputs, all_classes, train_size=0.7, random_state=1)

    return train_inputs, test_inputs, train_classes, test_classes


def MLP_classifier(train_data, train_labels, hidden_layer_sizes, activation_function):
    clf = MLPClassifier(solver='lbfgs', 
                        alpha=1e-5,
                        hidden_layer_sizes=hidden_layer_sizes, 
                        random_state=1,
                        max_iter=500,
                        activation=activation_function)

    clf.fit(train_data, train_labels) 
    return clf


def plot_confusion_matrix(test_classes, predictions, labels, ax=None, title=None):
    cm = confusion_matrix(test_classes, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    if title and ax is not None:
        ax.set_title(title)
    return disp

def main():
    train_inputs, test_inputs, train_classes, test_classes = load_diabetes_dataset()


    # hidden_layer_sizes = [(6,3,), (6, 6,), (10, 15, 6,)]
    # activation_functions = ['identity', 'logistic', 'tanh', 'relu']


    hidden_layer_sizes = [(6, 3), (6, 6)]
    activation_functions = ['relu']

    confusion_matrix_number = len(activation_functions) * len(hidden_layer_sizes)

    fig, axes = plt.subplots(1, confusion_matrix_number, figsize=(6 * confusion_matrix_number, 5))
    if confusion_matrix_number == 1:
        axes = [axes]

    i = 0
    for activation_function in activation_functions:
        print('--------------------------------')
        print(f"Activation function: {activation_function}")
        for hidden_layer_size in hidden_layer_sizes:
            clf = MLP_classifier(train_inputs, train_classes, hidden_layer_size, activation_function)
            predict_for_test_inputs = clf.predict(test_inputs)
            predict_for_train_inputs = clf.predict(train_inputs)
            print(f"Train accuracy (for {hidden_layer_size}) train_inputs: ", accuracy_score(train_classes, predict_for_train_inputs))
            print(f"Train accuracy (for {hidden_layer_size}) test_inputs: ", accuracy_score(test_classes, predict_for_test_inputs))

            plot_confusion_matrix(test_classes, predict_for_test_inputs, sorted(list(set(test_classes))), ax=axes[i], title=f"Confusion matrix for {activation_function} and {hidden_layer_size}")
            i += 1


    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()