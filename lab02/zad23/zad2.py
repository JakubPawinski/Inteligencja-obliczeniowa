import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np



FEATURES = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

def analyze_pca_variance_loss(df):
    x = df.loc[:, FEATURES].values
    y = df.loc[:, ['variety']].values
    x = StandardScaler().fit_transform(x)

    pca_full = PCA(n_components=len(FEATURES))
    pca_full.fit(x)


    print("PCA_full for all features:")
    print(pca_full.explained_variance_ratio_)


    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    print("\nCumulative variance:")
    print(cumulative_variance)

    print("\nVariance loss after removing last i columns:")
    for i in range(1, 4):
        loss = sum(pca_full.explained_variance_ratio_[4-i:]) / sum(pca_full.explained_variance_ratio_)
        print(f"Loss after removing {i} last columns: {loss:.4f} which is {loss*100:.2f}%")

    
    min_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nMinimum number of components to retain 95% variance: {min_components}")


def calculate_pca(df, components_number):
    x = df.loc[:, FEATURES].values
    y = df.loc[:, ['variety']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=components_number)
    principalComponents = pca.fit_transform(x)
    print("\nPCA for 3 features:")
    print(pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(data=principalComponents, 
                            columns=[f'principal component {i}' for i in range(1, components_number+1)])
    finalDf = pd.concat([principalDf, df[['variety']]], axis=1)
    
    return finalDf


def show_plot(finalDf):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)


    targets = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['variety'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                finalDf.loc[indicesToKeep, 'principal component 2'],
                c=color,
                s=50)
    ax.legend(targets)
    ax.grid()

    plt.show() 

def show_plot_3d(finalDf):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    ax.set_title('3 component PCA', fontsize=16)

    targets = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['r', 'g', 'b']
    
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['variety'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'], 
                   c=color,
                   s=50)
    
    ax.legend(targets)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("iris1.csv")

    analyze_pca_variance_loss(df)

    finalDf = calculate_pca(df, 2)
    show_plot(finalDf)
    # show_plot_3d(finalDf)

if __name__ == "__main__":
    main()