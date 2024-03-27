import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
from Kmeans import Kmeans

class Synthetic_data:
    def __init__(self,initial_data):
        self.synthetic_data = self.generate_data(initial_data)
        
    def generate_data(self,initial_data):
            # Assuming 'data' is your original dataset as a 2D NumPy array
        min_values = initial_data.min(axis=0)  # Minimum values for each column/feature
        max_values = initial_data.max(axis=0)  # Maximum values for each column/feature
        np.random.seed(314159)
        # Generate synthetic data
        synthetic_data = np.random.uniform(low=min_values, high=max_values, size=initial_data.shape)
        #print(type(synthetic_data))
        #print(synthetic_data)
        Synthetic_df= pd.DataFrame(data=synthetic_data[1:,1:],    # values
                               index=synthetic_data[1:,0],    # 1st column as index
                               columns=synthetic_data[0,1:])
        #Synthetic_df.to_csv("synth.csv")
        return Synthetic_df


if __name__ == "__main__":
   
    
    # Initialize the Kmeans class with the path to your dataset file
   kmeans_instance = Kmeans('dataset')
   #print(kmeans_instance.data)
   synthetic_data_instance = Synthetic_data(kmeans_instance.data)
   #print(synthetic_data_instance.synthetic_data)
   synthetic_data = synthetic_data_instance.generate_data(kmeans_instance.data)
   #print(synthetic_data)
   #print(type(synthetic_data))
   # Select the number of clusters (k) and the maximum number of iterations for the k-means algorithm
   k = 5
   max_iterations = 100

   # Perform the clustering with the specified number of clusters and iterations
   # This will return the final centroids of the clusters
   centroids = kmeans_instance.clustername(synthetic_data, k, max_iterations)
   #print(centroids)
   
   # Assign cluster IDs to each data point based on the final centroids
   # This step is often integrated into the clustering process but is shown here for clarity
   cluster_ids = kmeans_instance.assignClusterIds(synthetic_data, centroids)
   #print(cluster_ids)
   # Compute the silhouette score for the clustering
   silhouette_score = kmeans_instance.compute_silhouette(synthetic_data, centroids)
   #print(silhouette_score)
   #print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

   # Plot silhouette scores for a range of k values to find the optimal number of clusters
   kmeans_instance.plot_silhouette(k_range=range(2, 10))  # Example range from 2 to 10
   print(f"Silhouette score for k={k}: {silhouette_score:.4f}")


    
    