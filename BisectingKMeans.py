#import relevant libraries
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from Kmeans import Kmeans


class BisectingKmeans:
    def __init__(self,input_file):
            self.data = input_file
            
        
    @staticmethod
    def FileConversion(input_file):
        df = pd.read_csv(input_file, header=None, sep="[\t ]+", engine='python')
        df = df.iloc[:, 1:] 
        return df.to_numpy()
        

    def computeDistance(self,a,b):
        dist = np.linalg.norm(a-b)
        return dist

    def initialSelection(self,data,k):
        np.random.seed(314159)
        centroids = [data]
        return centroids 
        
        
    def clustername(self,data,k,maxIter):
        clusters = [data]
       
        
        # Initialize a single cluster containing indices of all data points
        clusters_indices = [np.arange(len(data))]

        while len(clusters_indices) < k:
        # Select the cluster to split based on maximum SSE
            sse_list = [self.computeSumfSquare(data[indices]) for indices in clusters_indices]
            idx_to_split = np.argmax(sse_list)
            indices_to_split = clusters_indices.pop(idx_to_split)

        # Perform K-means clustering with k=2 on the selected cluster
            cluster_to_split = data[indices_to_split]
            kmeans_instance = Kmeans(cluster_to_split)
            _, labels = kmeans_instance.clustername(cluster_to_split, 2, maxIter)

        # Split the indices based on labels and add them as new clusters
            indices1 = indices_to_split[labels == 0]
            indices2 = indices_to_split[labels == 1]
            clusters_indices.append(indices1)
            clusters_indices.append(indices2)

        # Assign labels based on cluster indices
        final_labels = np.empty(len(data), dtype=int)
        for label, indices in enumerate(clusters_indices):
            final_labels[indices] = label

        # Calculate final centroids for each cluster
        final_centroids = [np.mean(data[indices], axis=0) for indices in clusters_indices]

        #print(np.array(final_centroids),final_labels)
        return np.array(final_centroids), final_labels

    def computeSumfSquare(self,data):
        centroid = np.mean(data, axis=0)
        return np.sum(np.square(data - centroid))
       

    def compute_silhouette(self,labels,k):
       # print(labels.shape)
        #print(type(labels[0]))
        #print(k)
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



    
    @staticmethod
    def plot_silhouette(k_range, silhouette_scores):
        plt.plot(k_range, silhouette_scores, "-bo")
        plt.xlabel('k')
        plt.ylabel('Silhouette Coefficient')
        plt.title('Silhouette Coefficient for Bisecting K-means')
        plt.xticks(list(k_range))  # Ensure ticks match k values
        plt.show()
        
    #x = FileConversion(self, 'dataset')
if __name__ == '__main__':
       data = BisectingKmeans.FileConversion('dataset')  # Adjust path as needed
       silhouette_scores = []
       k_range = range(2, 11)  # Example range

       for k in k_range:
            bisectingkmeans_instance = BisectingKmeans(data)
            maxIter = 100
            centroids, labels = bisectingkmeans_instance.clustername(data, k, maxIter)

        # Compute the silhouette score for the current k
            silhouette_score = bisectingkmeans_instance.compute_silhouette(labels, k)
            silhouette_scores.append(silhouette_score)
            print(f"Silhouette score for k={k}: {silhouette_score:.4f}")
            
        # Plot silhouette scores for a range of k values to find the optimal number of clusters
       BisectingKmeans.plot_silhouette(k_range,silhouette_scores)  # Example range from 2 to 10