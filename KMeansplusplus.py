import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


class Kmeansplusplus:
    def __init__(self, data):
        self.data = np.array(data)

    @staticmethod
    def FileConversion(input_file):

        df = pd.read_csv(input_file, header=None,
                         sep="[\t ]+", engine='python')
        df = df.iloc[:, 1:]
        return df.to_numpy()

    def computeDistance(self, a, b):
        dist = np.linalg.norm(a-b)
        return dist

    def initialSelection(self, data, k):
        # print(data)
        # print(k)
        np.random.seed(314159)
        #datatest1 = data.to_numpy()
        # initial_centroid = np.random.uniform(np.amin(data, axis = 0), np.amax(data, axis=0),
        #                                    size=data.shape[1])
        # print(type(self.data.dtype))
        initial_centroid = np.random.uniform(np.amin(self.data, axis=0), np.amax(self.data, axis=0),
                                             size=self.data.shape[1])
        #print(initial_centroid)
       # print(initial_centroid)
        # print(len(initial_centroid))
        centroids = [initial_centroid]

        # print(centroids)
        #print(len(centroids))
        #centroids = np.zeros((k, data.shape[1]))
        #print(k)
        for i in range(k):
            #print(k)
            # print("before")
            distances = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in self.data])
            # print("after")
            # Compute the probabilities for each point
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)

            # Select the next centroid
            r = random.uniform(0, 1)
            #index = np.where(cumulative_probabilities >= r)[0][0]
            # centroids.append(datatest1[index])
            for idx, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(self.data[idx])  # Use idx here
                    break
        # print(centroids)
        centroids = np.array(centroids)

        #print("Initial centroids:\n", centroids)
        return centroids


    def assignClusterIds(self, data, centroids):
        data_array = np.array(self.data)
        distance_matrix = []
       # print(len(centroids))
    # Loop through each data point and calculate distances to each centroid
        for i, data_point in enumerate(data_array):
            dist = []
           # print(f"Data Point {i}: {data_point}")  # Print the current data point

        # Calculate distance from this point to each centroid
            for j, centroid in enumerate(centroids):
                distance = self.computeDistance(data_point, centroid)
                dist.append(distance)
                #print(f"    Distance to Centroid {j}: {distance:.4f}")  # Print the distance to each centroid

        # Add this point's distances to all centroids to the distance matrix
            distance_matrix.append(dist)

    # Convert the list of lists into a NumPy array
        distance_matrix = np.array(distance_matrix)

        clusters = np.argmin(distance_matrix, axis=1)


        return clusters

    def computeClusterRepresentatives(self, data, cluster_ids,centroids):
        unique_clusters = np.unique(cluster_ids)
        print(f"Unique cluster IDs: {unique_clusters}")
        print(f"Number of unique clusters: {len(unique_clusters)}")
        new_centroids = []
        if len(unique_clusters) < len(centroids):
            print(f"Some original centroids have no points assigned to them.")
        for cluster_id in unique_clusters:
            # Get all data points assigned to the current cluster
            cluster_points = data[cluster_ids == cluster_id]

        # Calculate the mean position of these points
            centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(centroid)

    # Convert the list of new centroids into a NumPy array
        new_centroids = np.array(new_centroids)
        #print(new_centroids)
        return new_centroids

    def clustername(self, k, maxIter):
        centroids = self.initialSelection(self.data, k)
        print("Initial centroids:\n", centroids)

        for i in range(maxIter):
            print(f"Iteration {i+1}")

            labels = self.assignClusterIds(self.data, centroids)
            print(f"Cluster assignments for iteration {i+1}: {np.unique(labels, return_counts=True)}")
            print(labels)
            new_centroids = self.computeClusterRepresentatives(self.data, labels,centroids)
            print(f"New centroids for iteration {i+1}:\n{new_centroids}")

            
            centroids = new_centroids
            print(type(new_centroids))
            print(f"new centroid list is {centroids}")
            are_equal = np.all(np.isclose(centroids, new_centroids, atol=1e-08))
            if are_equal:
                print("Convergence reached.")
                break
        final_labels = self.assignClusterIds(self.data, centroids)
        return centroids, final_labels

    def compute_silhouette(self,labels):
        # print(labels.shape)
         #print(type(labels[0]))
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

    def plot_silhouette(self, k_range=range(2, 10)):
        silhouette_scores_kmeans = []
        for k in k_range:
            labels1 = self.clustername(k, 1)
            score1 = self.compute_silhouette(labels1)

            print(
                f"Silhouette coefficient for k={k} with K-means: {score1:.4f}")

            silhouette_scores_kmeans.append(score1)

        plt.plot(range(2, 10), silhouette_scores_kmeans, "-bo")
        plt.xlabel('k')
        plt.ylabel('Silhouette Coefficient')
        plt.title('Silhouette Coefficient for K-means ++')
        plt.show()

    #x = FileConversion(self, 'dataset')
if __name__ == '__main__':
    data = Kmeansplusplus.FileConversion('dataset')
    kmeanspp_instance = Kmeansplusplus(data)
    #print(k)
    k = 3
    maxIter = 1
    #print(k)
    _, labels = kmeanspp_instance.clustername(k, maxIter)

    # Compute the silhouette score for the clustering
    silhouette_score = kmeanspp_instance.compute_silhouette(labels)
    #print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for a range of k values to find the optimal number of clusters
    kmeanspp_instance.plot_silhouette(
        k_range=range(2, 10))  # Example range from 2 to 10


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
