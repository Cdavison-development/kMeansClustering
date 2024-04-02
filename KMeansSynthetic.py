import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
from Kmeans import Kmeans

class Synthetic_data:
    def __init__(self, data):
        self.data = data
        
    def generate_synthetic_data(self):
        np.random.seed(314159)
        
        min_vals = self.data.min(axis=0)
        max_vals = self.data.max(axis=0)
        synthetic_data = np.random.uniform(min_vals, max_vals, (self.data.shape[0] , self.data.shape[1]))
        #synthetic_data = original_data
        return synthetic_data

def plot_silhouette_scores(k_values, silhouette_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
    plt.title('Silhouette Scores for Different Values of k')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Step 1: Load original dataset
    original_data = Kmeans.load_dataset('dataset')  # Adjust the path to your dataset file

    # Step 2: Generate synthetic data of the same size as the original dataset
    synthetic_data_generator = Synthetic_data(original_data)
    synthetic_data = synthetic_data_generator.generate_synthetic_data()

    # Step 3: Run K-Means on the synthetic data
    kmeans_instance = Kmeans(synthetic_data)
    k_values = range(1, 10)
    maxIter = 100
    #centroids, labels = kmeans_instance.clustername(synthetic_data, k, maxIter)

    # Initialize a list to store silhouette scores
    silhouette_scores = []

    # Iterate through the range of k values
    for k in k_values:
        # Run K-Means clustering for the current k
        centroids, labels = kmeans_instance.clustername(synthetic_data, k, maxIter)
        # Compute and store the silhouette score for the current k
        silhouette_score = kmeans_instance.compute_silhouette(labels, k)
        silhouette_scores.append(silhouette_score)
        print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for all k values
    plot_silhouette_scores(k_values, silhouette_scores)
    
    