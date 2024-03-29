import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


class Kmeansplusplus:
    def __init__(self,input_file):
        self.data = self.FileConversion(input_file)
        
    def FileConversion(self,input_file):
        

        # The path to your output CSV file
        output_file = 'converted_file.csv'

        # Open the input file in read mode and the output file in write mode
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            # Create a CSV writer object for the output file
            csv_writer = csv.writer(outfile)

        # Read each line from the input file
            for line in infile:
            # Split the line into parts using space as the delimiter
                parts = line.strip().split(' ')
                csv_writer.writerow(parts)
        df = pd.read_csv(output_file, header=None)
        #df.columns = range(df.shape[1])
        numeric_data = df.iloc[:, 1:]
        #print(numeric_data.head())
        #numeric_data.to_csv("numeric.csv")
        #print(numeric_data)
        return numeric_data
        

    def computeDistance(self,a,b):
        dist = np.linalg.norm(a-b)
        return dist

    def initialSelection(self,data,k):
        #print(data)
        #print(k)
        np.random.seed(314159)
        datatest1 = data.to_numpy()
        #initial_centroid = np.random.uniform(np.amin(data, axis = 0), np.amax(data, axis=0),
         #                                    size=data.shape[1])
        initial_centroid = np.random.uniform(np.amin(datatest1, axis = 0), np.amax(datatest1, axis=0),
                                             size=datatest1.shape[1])
        #print(len(initial_centroid))
        centroids = [initial_centroid]
        #print(centroids)
        #print(initial_centroid)
        #centroids = np.zeros((k, data.shape[1]))
   
        for i in range(k-1):
            #print("before")
            distances = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in datatest1])
            #print("after")
            # Compute the probabilities for each point
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            
            # Select the next centroid
            r = random.uniform(0, 1)
            #index = np.where(cumulative_probabilities >= r)[0][0]
            #centroids.append(datatest1[index])
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    i = j
                    break
            centroids.append(datatest1[i])
        #print(centroids)
        centroids = np.array(centroids)
        
        #centroids.to_csv("centroids.csv")
       # print(centroids)
        return np.array(centroids)
   
    
        """
        #print(type(initial_centroid))
        data_array = np.array(data)
        distances = []
        for data_point in data_array:
            distance = self.computeDistance((np.array(data_point), np.array(centroid_list)) ** 2)
            distances.append(distance)
            
        #print(type(distances))
        print(len(data))
        #probabilities = []
        distancesTotal = sum(distances)
        for i in distances:
            probability = distances / distancesTotal
            cumulative_probability = probability.cumsum()
            #print(cumulative_probability)
            r = random.uniform(0, 1)
            for j, p in enumerate(cumulative_probability):
                if p > r:
                    l = j
                    print(p)
                    break
            centroid_list.append(data[l])
        
        centroid_list = np.array(centroid_list)   
        
        
        return centroid_list 
    """
    def assignClusterIds(self,data,centroids):
    
        data_array = np.array(data)
    
        c_list = []
        
        #print("test")
        for centroid in centroids:
            dist = []
            #print(np.array(centroid).shape)
            #print(len(centroid))
            #print(centroid)
            for data_point in data_array:
                #if len(np.array(centroid)) != 300:
                    
                    #print(centroid)
                #print(len(np.array(data_point)), len(np.array(centroid)))
                distance = self.computeDistance(np.array(data_point), np.array(centroid))
                dist.append(distance)
            
        
            c_list.append(dist)
        
        
        df = pd.DataFrame(np.array(c_list)).transpose()

        clusters = df.idxmin(axis=1)  # This creates a new DataFrame with two columns: the index and the cluster IDs
    
        return clusters
    

    def computeClusterRepresentatives(self,data, cluster_ids):
        #print(cluster_ids.values)
        # Ensure the cluster IDs are in the same order as the original data points
        data_copy = data.copy()
        data_copy['ClusterID'] = cluster_ids.to_numpy()
        
        new_centroids = data_copy.groupby('ClusterID').mean()
   

       #print((new_centroids))
        centroids_array = new_centroids.values
   
        return centroids_array



    def clustername(self,data,k,maxIter):
       centroids = self.initialSelection(data, k)
       #print(data)
       for i in range(maxIter):
           C = self.assignClusterIds(data,centroids)
           new_centroids = self.computeClusterRepresentatives(data, C)
           centroids = new_centroids
           #print(centroids)
       #print(centroids)
       return centroids
       

    def compute_silhouette(self,data, clusters):
        data = np.array(data)
        #print(data)
        n = len(data)
        num_clusters = len(clusters)
        
        # Precompute all distances between data points and cluster centers
        distances = np.array([[np.linalg.norm(data_point - cluster_center) for cluster_center in clusters] for data_point in data])
        
        # Determine the nearest cluster for each data point
        nearest_clusters = np.argmin(distances, axis=1)
        
        silhouette_scores = []
        for i in range(n):
        # Identifying the cluster of the current point
            own_cluster = nearest_clusters[i]
        
        # Calculating average distance from point i to other points in the same cluster
            in_cluster_mask = (nearest_clusters == own_cluster) & (np.arange(n) != i)

# Use the mask to select distances to the own cluster center for those points
            if np.any(in_cluster_mask):
                a_i = np.mean(distances[in_cluster_mask, own_cluster])
            else:
                a_i = 0  # Default to 0 if point i is the only member of its cluster
        
            point_distances = np.linalg.norm(data[i] - data, axis=1)
        # Calculating the smallest average distance to all points in each other cluster
            b_i = np.inf
            for other_cluster in range(num_clusters):
                if other_cluster != own_cluster:
        # Create a mask for selecting data points in the other cluster
                    other_cluster_mask = nearest_clusters == other_cluster
        
        # Use the mask to filter rows and then select the column corresponding to the other cluster's center
                    if np.any(other_cluster_mask):
                        other_cluster_distances = point_distances[other_cluster_mask]
                        other_cluster_avg = np.mean(other_cluster_distances)
                        b_i = min(b_i, other_cluster_avg)
        
        # Compute the silhouette score for point i
            if max(a_i, b_i) > 0:
                silhouette_scores.append((b_i - a_i) / max(a_i, b_i))
            else:
                silhouette_scores.append(0)

    # Returning the mean silhouette score
        score = np.mean(silhouette_scores)
        return score



    
    def plot_silhouette(self,k_range=range(2,10)):
        silhouette_scores_kmeans = []
        for k in k_range:
            labels1 = self.clustername(self.data, k, 100)
            score1 = self.compute_silhouette(self.data, labels1)

            print(f"Silhouette coefficient for k={k} with K-means: {score1:.4f}")

            silhouette_scores_kmeans.append(score1)


        plt.plot(range(2, 10), silhouette_scores_kmeans, "-bo")
        plt.xlabel('k')
        plt.ylabel('Silhouette Coefficient')
        plt.title('Silhouette Coefficient for K-means ++')
        plt.show()
        
    #x = FileConversion(self, 'dataset')
if __name__ == '__main__':
    kmeanspp_instance = Kmeansplusplus('dataset')
    
    k=9
    maxIter = 100
    centroids = kmeanspp_instance.clustername(kmeanspp_instance.data, k, maxIter)

    # Assign cluster IDs to each data point based on the final centroids
    # This step is often integrated into the clustering process but is shown here for clarity
    cluster_ids = kmeanspp_instance.assignClusterIds(kmeanspp_instance.data, centroids)

    # Compute the silhouette score for the clustering
    silhouette_score = kmeanspp_instance.compute_silhouette(kmeanspp_instance.data, centroids)
    #print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for a range of k values to find the optimal number of clusters
    kmeanspp_instance.plot_silhouette(k_range=range(2, 10))  # Example range from 2 to 10


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

