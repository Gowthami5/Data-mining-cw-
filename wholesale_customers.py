# Part 2: Cluster Analysis

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from itertools import combinations


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    # read csv data from data_file
    df = pd.read_csv(data_file, index_col=False, skipinitialspace=True)
    # drop 'Channel', 'Region'
    df = df.drop(['Channel', 'Region'], axis=1)
    # replace any null '?' characters
    df.replace(to_replace=[r' ?', r'?', r'? ', r' ', r''],
               value=[np.nan, np.nan, np.nan, np.nan, np.nan], regex=False, inplace=True)

    return df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    # create label for statistics
    index_labels = ['mean', 'std', 'min', 'max']
    # dataframe for summary statistics
    df_summary = pd.DataFrame([round(df.mean()).astype(int), round(df.std()).astype(int),
                               round(df.min()).astype(int), round(df.max()).astype(int)],
                              index=index_labels)

    return df_summary


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    # get statistics using summary_statistics function
    df_summary = summary_statistics(df)
    # run through each column and standardize the column using the stats values
    for column in df.columns:
        df[column] = (df[column] - df_summary[column]['mean']) / df_summary[column]['std']

    return df


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    # assignment of instances to clusters, using kmeans
    kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300)
    kmeans.fit(df)
    # return labels as a pandas series
    y = pd.Series(kmeans.labels_)

    return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    pass


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    # assignment of instances to clusters, using agglomerative hierarchical clustering
    agglomerative = AgglomerativeClustering(n_clusters=k)
    agglomerative.fit(df)
    # return labels as a pandas series
    y = pd.Series(agglomerative.labels_)

    return y


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X, y):
    silhouette_scores = silhouette_score(X, y)

    return silhouette_scores


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    # define the cluster array k
    k = (3, 5, 10)

    # holder for results
    algoEval_df = []

    # 'k mean clustering'
    for k_i in k:
        for i in range(10):
            y = kmeans(df, k_i)
            # print(y)
            scoreKmean = clustering_score(df, y)
            # print(scoreKmean)
            algoEval_df.append(
                {
                    'Algorithm': 'Kmeans',
                    'data': 'Original',
                    'k': k_i,
                    'Silhouette Score': scoreKmean
                }
            )

    # 'agglomerative hierarchical clustering'
    for k_i in k:
        c = agglomerative(df, k_i)
        # print(c)
        scoreAgglomerative = clustering_score(df, c)
        # print(scoreAgglomerative)
        algoEval_df.append(
            {
                'Algorithm': 'Agglomerative',
                'data': 'Original',
                'k': k_i,
                'Silhouette Score': scoreAgglomerative
            }
        )

    # Next, standardize each attribute value by subtracting with the mean and then
    # dividing with the standard deviation for that attribute.
    dfstd = standardize(df)

    # Repeat the previous kmeans and agglomerative hierarchical clustering executions
    # with the standardized data set.

    # 'k mean clustering'
    for k_i in k:
        for i in range(10):
            y = kmeans(dfstd, k_i)
            # print(y)
            scoreKmean = clustering_score(dfstd, y)
            # print(scoreKmean)
            algoEval_df.append(
                {
                    'Algorithm': 'Kmeans',
                    'data': 'Standardized',
                    'k': k_i,
                    'Silhouette Score': scoreKmean
                }
            )

    # 'agglomerative hierarchical clustering'
    for k_i in k:
        c = agglomerative(dfstd, k_i)
        # print(c)
        scoreAgglomerative = clustering_score(dfstd, c)
        # print(scoreAgglomerative)
        algoEval_df.append(
            {
                'Algorithm': 'Agglomerative',
                'data': 'Standardized',
                'k': k_i,
                'Silhouette Score': scoreAgglomerative
            }
        )

    return pd.DataFrame(algoEval_df)


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    # evaluation dataframe
    eval_df = rdf.max()

    return eval_df['Silhouette Score']



# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):

    # generate clusers for the best model found
    # (Algorithm: Kmeans, data: Standardized, k: 10, Silhouette Score: 0.548289)
    dfstd = standardize(df)
    group = kmeans(dfstd, 10)

    fig, ax = plt.subplots(5, 3, figsize=(10, 10))

    # Get all permutations of and length 2
    comb = combinations(range(df.shape[1]), 2)

    plotIdxRow = [row for row in range(5)]
    plotIdxCol = [column for column in range(3)]
    plotIdx = 0

    for combi in list(comb):
        # print(combi)
        scatter_x = np.array(df.iloc[:, combi[0]])  # np.array([1,2,3,4,5])
        scatter_y = np.array(df.iloc[:, combi[1]])  # np.array([5,4,3,2,1])
        name = df.columns[combi[0]] + ' vs ' + df.columns[combi[1]]

        for g in np.unique(group):
            i = np.where(group == g)
            ax[plotIdxRow[plotIdx % 5]][plotIdx % 3].scatter(scatter_x[i], scatter_y[i], label=g)
            ax[plotIdxRow[plotIdx % 5]][plotIdx % 3].set_title(name, size=6)
        # ax[0][0].legend()

        plotIdx = plotIdx + 1

    fig.tight_layout(pad=1.0)
    plt.savefig("Kmean10plots.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# test Script
if __name__ == "__main__":

    # read the data
    df = read_csv_2('wholesale_customers.csv')

    # 1. [10 points] Compute the mean, standard deviation, minimum, and maximum value for each
    # attribute. Round the mean and standard deviation to the closest integers.
    df_summary = summary_statistics(df)
    print('Summary Statistics:\n', df_summary.head())

    # 2. [20 points] Divide the data points into k clusters, for k ∈ {3, 5, 10}, using
    # kmeans and agglomerative hierarchical clustering.
    # Because the performance of kmeans (e.g. number of iterations) is significantly affected
    # by the initial cluster center selection, repeat 10 executions of kmeans for each k value.
    score_df = cluster_evaluation(df)
    print('Cluster Evaluation:\n', score_df.head(65))

    # Identify which run resulted in the best set of clusters using the Silhouette score
    # as your evaluation metric.
    bestScore = best_clustering_score(score_df)
    print('Best Clustering Score:\n', bestScore)

    # Visualize the best set of clusters computed in the previous question.
    # For this, construct a scatterplot for each pair of attributes using Pyplot.
    # Therefore, 15 scatter plots should be constructed in total.
    # Different clusters should appear with different colors in each scatter plot. Note that these plots
    # could be used to manually assess how well the clusters separate the data points.
    scatter_plots(df)