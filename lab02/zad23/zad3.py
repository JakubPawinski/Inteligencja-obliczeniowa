import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FEATURES = ['sepal.length', 'sepal.width']

def zscore(df):
    df_z = df.copy()
    zscore_scaler = StandardScaler()
    df_z[FEATURES] = zscore_scaler.fit_transform(df_z[FEATURES])
    return df_z

def minmax(df):
    df_mm = df.copy()
    minmax_scaler = MinMaxScaler()
    df_mm[FEATURES] = minmax_scaler.fit_transform(df_mm[FEATURES])
    return df_mm

def plot_data(ax, df, title):
    sns.scatterplot(x=df[FEATURES[0]], y=df[FEATURES[1]], hue=df['variety'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(FEATURES[0])
    ax.set_ylabel(FEATURES[1])

def main():
    df = pd.read_csv("iris1.csv")

    df_z = zscore(df)
    df_mm = minmax(df)


    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    plot_data(axs[0], df, "Original data")
    plot_data(axs[1], df_z, "Z-score")
    plot_data(axs[2], df_mm, "MinMax")
    

    plt.show()

if __name__ == "__main__":
    main()