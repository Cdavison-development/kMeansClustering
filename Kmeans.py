#import libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd



#define Kmeans class
class Kmeans:
    
    """
    input(s) : data
    output: None
    
    Initialises an instance for Kmeans class
    Converts data to np array
    
    """
    def __init__(self, data):
        
        self.data = np.array(data)
        np.random.seed(314159)
        random.seed(3)
    
    """
    input(s) : dataset file  
    output: 
        df.to_numpy(): dataframe converted to numpy array
    
    Converts the 'dataset' file to csv and reads it as a dataframe,
    removes first column of the dataframe and converts the dataframe to numpy array
    """        
    @staticmethod
    def load_dataset(input_file):
        
        df = pd.read_csv(input_file, header=None, sep="[\t ]+", engine='python')
        df = df.iloc[:, 1:] 
        return df.to_numpy()
        
    """
    input(s) : 
        a,b : numpy arrays representing points in a dataset
    output: 
        dist: value representing distance between two points 'a' and 'b'
    
    This function calculates the euclidean distance between points 'a' and 'b'
    and returns the distance between the points
    """
    def computeDistance(self,a,b):
        dist = np.linalg.norm(a-b)
        return dist

    """
    input(s) : 
        data: numpy array storing dataset data
        k: integer representing number of clusters to be used
    output: 
        centroids: numpy array of shape (K, n), where k is the number of clusters and
        n is number features in the dataset
    
    initialises k centroids for clustering, it selects the first centroid randomly and
    every following centroid based on distance from existing centroids, attemping to evenly 
    spread out the centroids.
    """
    def initialSelection(self,data,k):
        centroids = np.random.uniform(np.amin(self.data, axis = 0), np.amax(self.data, axis=0), 
                                  size = (k, self.data.shape[1]))
        return centroids 
    
    """
    input(s):
        data: numpy array storing dataset data
        centroids: numpy array of current centroids, one for each cluster
        
    output:
        clusters: numpy array where each element is the index of the nearest centroid
    """
    def assignClusterIds(self,data,centroids):
    
        data_array = np.array(self.data)
        distance_matrix = []
            
        for data_point in data_array:
            dist = []

            for centroid in centroids:
                distance = self.computeDistance(data_point, centroid)  
                dist.append(distance)


            distance_matrix.append(dist)
        distance_matrix = np.array(distance_matrix)
        clusters = np.argmin(distance_matrix, axis=1)
       
        return clusters
    

    def computeClusterRepresentatives(self,data, cluster_ids):
        unique_clusters = np.unique(cluster_ids)
        new_centroids = []

        for cluster_id in unique_clusters:
            cluster_points = self.data[cluster_ids == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)

        new_centroids = np.array(new_centroids)
        return new_centroids
   



    def clustername(self,data,k,maxIter):
       centroids = self.initialSelection(data, k)
       for i in range(maxIter):
           labels = self.assignClusterIds(data,centroids)
           new_centroids = self.computeClusterRepresentatives(data, labels)
           centroids = new_centroids
       final_labels = self.assignClusterIds(data, centroids)
       return centroids, final_labels
       

    def compute_silhouette(self,labels,k):
       
        if k==1:
            silhouette_coefficient=0
        else:
            distances = np.sqrt(((self.data[:, np.newaxis, :] - self.data[np.newaxis, :, :])**2).sum(axis=2))

            a = np.zeros(len(self.data))
            b = np.inf * np.ones(len(self.data))
            for i in range(len(self.data)):
                a[i] = np.mean(distances[i, labels == labels[i]])
                for cluster in set(labels):
                    #print(labels)
                    if cluster != labels[i]:
                        cluster_mask = (labels == cluster)
                        b[i] = min(b[i], np.mean(distances[i][cluster_mask]))
               
            s = (b - a) / np.maximum(a, b)
        
            silhouette_coefficient = np.mean(s)
        
        return silhouette_coefficient
   

    
    def plot_silhouette(self,k_values):
        silhouette_scores_kmeans = []
        for k in k_values:
            _, labels = self.clustername(self.data,k, 1)
            score = self.compute_silhouette(labels,k)
            silhouette_scores_kmeans.append(score)


        plt.figure(figsize=(8, 5))  # Set the figure size for better readability
        plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')  # Marker added for clarity
        plt.title('Silhouette Scores for Different k Values')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.xticks(k_values)  # Ensure all k values are marked
        plt.grid(True)  # Add a grid for better readability
        plt.show()

if __name__ == '__main__':
    data = Kmeans.load_dataset('dataset')  # No need for 'self' here
    kmeans_instance = Kmeans(data)
    
    k_values=range(1, 10)
    maxIter = 100
    silhouette_scores = []

    # Iterate through the range of k values
    for k in k_values:
        # Run K-Means clustering for the current k
        centroids, labels = kmeans_instance.clustername(data, k, maxIter)
        # Compute and store the silhouette score for the current k
        silhouette_score = kmeans_instance.compute_silhouette(labels, k)
        silhouette_scores.append(silhouette_score)
        print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for all k values
    kmeans_instance.plot_silhouette(k_values)


