import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd




class Kmeans:
    def __init__(self, data):
        
        self.data = np.array(data)
            
    @staticmethod
    def FileConversion(input_file):
        
        df = pd.read_csv(input_file, header=None, sep="[\t ]+", engine='python')
        df = df.iloc[:, 1:]  # Assuming you want to skip the first column
        return df.to_numpy()
        

    def computeDistance(self,a,b):
        dist = np.linalg.norm(a-b)
        return dist

    def initialSelection(self,data,k):
        #print(data)
        np.random.seed(314159)
        centroids = np.random.uniform(np.amin(self.data, axis = 0), np.amax(self.data, axis=0), 
                                  size = (k, self.data.shape[1]))
        #print(len(centroids))
        return centroids 
    
    def assignClusterIds(self,data,centroids):
    
        data_array = np.array(self.data)
    
        #c_list = []
        distance_matrix = []
        # Iterate over each data point
        for data_point in data_array:
            dist = []

        # Calculate distance from this point to each centroid
            for centroid in centroids:
                distance = self.computeDistance(data_point, centroid)  # No need to convert to np.array here if they're already arrays
                dist.append(distance)

        # Add this point's distances to all centroids to the matrix
            distance_matrix.append(dist)
            
    # Convert the list of lists into a NumPy array
        distance_matrix = np.array(distance_matrix)
        
    # Find the index of the minimum distance for each point
        clusters = np.argmin(distance_matrix, axis=1)
        #print(clusters)
        return clusters
    

    def computeClusterRepresentatives(self,data, cluster_ids):
        unique_clusters = np.unique(cluster_ids)
        new_centroids = []

        for cluster_id in unique_clusters:
        # Get all data points assigned to the current cluster
            cluster_points = self.data[cluster_ids == cluster_id]

        # Calculate the mean position of these points
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)

    # Convert the list of new centroids into a NumPy array
        new_centroids = np.array(new_centroids)

        return new_centroids
   



    def clustername(self,k,maxIter):
       centroids = self.initialSelection(self.data, k)
       #print(data)
       for i in range(maxIter):
           labels = self.assignClusterIds(self.data,centroids)
           new_centroids = self.computeClusterRepresentatives(self.data, labels)
           centroids = new_centroids
           #print(centroids)
       #print(centroids)
       final_labels = self.assignClusterIds(self.data, centroids)
       return centroids, final_labels
       

    def compute_silhouette(self,labels):
        if k==1:
            silhouette_coefficient=0
        else:
            distances = np.sqrt(((data[:, np.newaxis, :] - data)**2).sum(axis=2))
            s = np.zeros(len(data))
            for i in range(len(data)):
                a_i = np.mean(distances[i, labels == labels[i]])
                b_i = np.min(np.mean(distances[i, labels != labels[i]]))
                s[i] = (b_i - a_i) / max(a_i, b_i)
            silhouette_coefficient = np.mean(s)
        return silhouette_coefficient
    


    
    def plot_silhouette(self,k_range=range(2,10)):
        silhouette_scores_kmeans = []
        for k in k_range:
            _, labels1 = self.clustername(k, 100)
            score1 = self.compute_silhouette(labels1)

            print(f"Silhouette coefficient for k={k} with K-means: {score1:.4f}")

            silhouette_scores_kmeans.append(score1)


        plt.plot(range(2, 10), silhouette_scores_kmeans, "-bo")
        plt.xlabel('k')
        plt.ylabel('Silhouette Coefficient')
        plt.title('Silhouette Coefficient for K-meansy')
        plt.show()
        
    #x = FileConversion(self, 'dataset')
if __name__ == '__main__':
    data = Kmeans.FileConversion('dataset')  # No need for 'self' here
    kmeans_instance = Kmeans(data)
    
    k=3
    maxIter = 100
    centroids, labels = kmeans_instance.clustername(k, maxIter)
    
    # Assign cluster IDs to each data point based on the final centroids
    # This step is often integrated into the clustering process but is shown here for clarity
    #cluster_ids = kmeans_instance.assignClusterIds(kmeans_instance.data, centroids)

    # Compute the silhouette score for the clustering
    silhouette_score = kmeans_instance.compute_silhouette(labels)
    #print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for a range of k values to find the optimal number of clusters
    kmeans_instance.plot_silhouette(k_range=range(2, 10))  # Example range from 2 to 10


"""
def compute_silhouette(x, clusters):
    n =len(x)
    
    x_array = np.array(x)
    
    distances = np.zeros((n, len(clusters)))
    for (i, point) in enumerate(x_array):
        for (j, center )in enumerate(clusters):
            distances[i, j] = computeDistance(point, center)
    
    cluster_assignments = np.argmin(distances, axis=1)
    print(cluster_assignments)
    silhouette = []

    for i in range(n):
        cluster = cluster_assignments[i]
        other_clusters = set(range(len(clusters))) - {cluster}
        same_cluster_points = [j for j in range(n) if cluster_assignments[j] == cluster and i != j]
        if not same_cluster_points:
            silhouette.append(0)
            continue
        avg_same_cluster = np.mean([computeDistance(x_array[i], x_array[j]) for j in same_cluster_points])
        avg_other_clusters = [np.mean([computeDistance(x_array[i], x_array[j]) for j in range(n) if cluster_assignments[j] == other_cluster]) for other_cluster in other_clusters]
        min_avg_other_clusters = np.min(avg_other_clusters)
        silhouette.append((min_avg_other_clusters - avg_same_cluster) / max(avg_same_cluster, min_avg_other_clusters))

    #print(np.mean(silhouette))
    return np.mean(silhouette)
"""