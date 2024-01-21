import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def find_optimal_clusters(X, max_clusters=10):
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, kmeans_labels))

    return silhouette_scores

def plot_silhouette_scores(silhouette_scores, max_clusters=10):
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Clusters')
    plt.show()

def plot_clusters(data, labels, centers):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def generic_clustering(csv_path,column, max_clusters=10):
    df = pd.read_csv(csv_path)

    print("Original Data:")
    print(df.head())

    column_to_drop = column
    df = df.drop(column_to_drop, axis=1)

    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    processed_data = preprocessor.fit_transform(df)
    silhouette_scores = find_optimal_clusters(processed_data, max_clusters)
    plot_silhouette_scores(silhouette_scores, max_clusters)

    optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 to get the actual number of clusters

    kmeans_model = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(processed_data)

    silhouette_kmeans = silhouette_score(processed_data, kmeans_labels)

    print(f'\nBest Clustering Algorithm: K-Means')
    print(f'Optimal Number of Clusters: {optimal_num_clusters}')
    print(f'Best Silhouette Score: {silhouette_kmeans:.4f}')

    df['Cluster_Labels_KMeans'] = kmeans_labels
    print(df[['Cluster_Labels_KMeans']])

    # Plot the distribution of clusters
    plt.figure(figsize=(8, 4))
    plt.bar(np.unique(kmeans_labels), np.bincount(kmeans_labels))
    plt.xlabel('Cluster Label')
    plt.ylabel('Count')
    plt.title('Cluster Distribution')
    plt.show()

    plot_clusters(processed_data, kmeans_labels, kmeans_model.cluster_centers_)

"""# Clustring Un Labeled Data
# Name of the CSV
# Column to Drop
"""

Data_CSV = 'concrete_data.csv' #@param {type:"string"}
Drop_column = "Cement" #@param {type:"string"}
generic_clustering(Data_CSV,Drop_column)