import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# df.describe()

# df['petal.width'].plot.hist()
# plt.show()





def decision_tree(train_inputs, test_inputs, train_classes, test_classes, df):
    dtc = DecisionTreeClassifier()
    clf = dtc.fit(train_inputs, train_classes)
    score = dtc.score(test_inputs, test_classes)
    print(score)

    return clf

def plot_decision_tree(clf, df, ax=None):
    tree.plot_tree(clf, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], class_names=df['variety'].unique(), filled=True, ax=ax)
    # plt.show()

def plot_confusion_matrix(test_classes, predictions, df, ax=None):
    cm = confusion_matrix(test_classes, predictions, labels=df['variety'].unique())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['variety'].unique())
    disp.plot(ax=ax)
    # plt.show()


def main():
    df = pd.read_csv('iris1 1.csv')

    all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    all_classes = df['variety'].values

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

    dtc = decision_tree(train_inputs, test_inputs, train_classes, test_classes, df)



    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_decision_tree(dtc, df, ax=axes[0])

    predictions = dtc.predict(test_inputs)
    
    plot_confusion_matrix(test_classes, predictions, df, ax=axes[1])
    axes[0].set_title("Decision tree")
    axes[1].set_title("Confusion matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()