import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

def plot_confusion_matrix(test_classes, predictions, df, ax=None, title=None):
    cm = confusion_matrix(test_classes, predictions, labels=df['variety'].unique())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['variety'].unique())
    disp.plot(ax=ax)
    if title:
        ax.set_title(title)
    return disp

def naive_bayes(train_inputs, test_inputs, train_classes, test_classes):
    gnb = GaussianNB()
    clf = gnb.fit(train_inputs, train_classes)
    score = gnb.score(test_inputs, test_classes)
    print(f"Score for Naive Bayes: {score}")
    return clf


def k_neighbours(train_inputs, test_inputs, train_classes, test_classes, n):
    knn = KNeighborsClassifier(n_neighbors=n)
    clf = knn.fit(train_inputs, train_classes)
    score = knn.score(test_inputs, test_classes)
    print(f"Score for {n} neighbours: {score}")

    return clf

def main():
    df = pd.read_csv('iris1 1.csv')

    all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    all_classes = df['variety'].values

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)


    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    neighbours = [3, 5, 11]

    for i, n in enumerate(neighbours):
        knn = k_neighbours(train_inputs, test_inputs, train_classes, test_classes, n)
        predictions = knn.predict(test_inputs)
        plot_confusion_matrix(test_classes, predictions, df, ax=axes[i], title=f"Confusion matrix for {n} neighbours")

    plot_confusion_matrix(test_classes, naive_bayes(train_inputs, test_inputs, train_classes, test_classes).predict(test_inputs), df, ax=axes[3], title="Confusion matrix for Naive Bayes")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()