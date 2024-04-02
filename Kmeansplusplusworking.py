import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


class Kmeansplusplus:
    def __init__(self,input_file):
        self.data = self.load_dataset(input_file)
        np.random.seed(314159)
        random.seed(3)
    def load_dataset(self,input_file):
        

        # The path to your output CSV file
        output_file = 'converted_file.csv'

        # Open the input file in read mode and the output file in write mode
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:

            csv_writer = csv.writer(outfile)

            for line in infile:
                parts = line.strip().split(' ')
                csv_writer.writerow(parts)
        df = pd.read_csv(output_file, header=None)
        numeric_data = df.iloc[:, 1:]
        return numeric_data
        

    def computeDistance(self,a,b):
        dist = np.linalg.norm(a-b)
        return dist

    def initialSelection(self,data,k):

        
        data = data.to_numpy()
        
        initial_centroid = np.random.uniform(np.amin(data, axis = 0), np.amax(data, axis=0),
                                             size=data.shape[1])
        centroids = [initial_centroid]
   
        for i in range(k):
            distances = np.array([min([self.computeDistance(x, c) for c in centroids]) for x in data])
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = random.uniform(0, 1)
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    next_centroid = data[j]
                    break
            centroids.append(next_centroid)
        #centroids = np.array(centroids)
        return np.array(centroids)
   
    
    def assignClusterIds(self,data,centroids):
        
        data_array = np.array(data)
        c_list = []

        for centroid in centroids:
            dist = []
            for data_point in data_array:
                distance = self.computeDistance(np.array(data_point), np.array(centroid))
                dist.append(distance)
            
            c_list.append(dist)
            
        df = pd.DataFrame(np.array(c_list)).transpose()
        clusters = df.idxmin(axis=1)  
        return clusters
    

    def computeClusterRepresentatives(self,data, cluster_ids):
        data_copy = data.copy()
        data_copy['ClusterID'] = cluster_ids.to_numpy()
        
        new_centroids = data_copy.groupby('ClusterID').mean()
   
        centroids_array = new_centroids.values

        return centroids_array



    def clustername(self,data,k,maxIter):
       centroids = self.initialSelection(data, k)

       for i in range(maxIter):
           C = self.assignClusterIds(data,centroids)
           new_centroids = self.computeClusterRepresentatives(data, C)
           centroids = new_centroids
           
       return centroids
       

    def compute_silhouette(self,data, clusters):
        data = np.array(data)
        n = len(data)
        num_clusters = len(clusters)
        
        distances_to_centers = np.array([[np.linalg.norm(data_point - center) for center in clusters] for data_point in data])
        nearest_clusters = np.argmin(distances_to_centers, axis=1)

        a = np.zeros(n)
        b = np.inf * np.ones(n)

        for i in range(n):
            own_cluster = nearest_clusters[i]
    
            in_cluster_mask = nearest_clusters == own_cluster
            a[i] = np.mean(distances_to_centers[in_cluster_mask, own_cluster]) if np.sum(in_cluster_mask) > 1 else 0

            for other_cluster in range(num_clusters):
                if other_cluster != own_cluster:
                    other_cluster_mask = nearest_clusters == other_cluster
                    if np.any(other_cluster_mask):
                        b[i] = min(b[i], np.mean(distances_to_centers[i, other_cluster]))

        s = (b - a) / np.maximum(a, b)
        s[np.isnan(s)] = 0 
        mean_silhouette_score = np.mean(s)
        return mean_silhouette_score



    
    def plot_silhouette(self,k_range=range(1,10)):
        silhouette_scores_kmeans = []
        for k in k_range:
            labels1 = self.clustername(self.data, k, 100)
            score1 = self.compute_silhouette(self.data, labels1)

            print(f"Silhouette coefficient for k={k} with K-means: {score1:.4f}")

            silhouette_scores_kmeans.append(score1)


        plt.plot(range(1, 10), silhouette_scores_kmeans, "-bo")
        plt.xlabel('k')
        plt.ylabel('Silhouette Coefficient')
        plt.title('Silhouette Coefficient for K-means ++')
        plt.show()
        
    
if __name__ == '__main__':
    kmeanspp_instance = Kmeansplusplus('dataset')
    
    k=4
    maxIter = 100
    centroids = kmeanspp_instance.clustername(kmeanspp_instance.data, k, maxIter)

    # Assign cluster IDs to each data point based on the final centroids
    # This step is often integrated into the clustering process but is shown here for clarity
    #cluster_ids = kmeanspp_instance.assignClusterIds(kmeanspp_instance.data, centroids)

    # Compute the silhouette score for the clustering
    silhouette_score = kmeanspp_instance.compute_silhouette(kmeanspp_instance.data, centroids)
    #print(f"Silhouette score for k={k}: {silhouette_score:.4f}")

    # Plot silhouette scores for a range of k values to find the optimal number of clusters
    kmeanspp_instance.plot_silhouette(k_range=range(1, 10))  # Example range from 2 to 10

