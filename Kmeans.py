import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
# The path to your input file

def FileConversion():
        input_file = 'dataset'

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
        numeric_data.to_csv("numeric.csv")
        return numeric_data
        

def computeDistance(a,b):
        dist = np.linalg.norm(a-b)
        return dist

def initialSelection(x,k):
    
        np.random.seed(314159)
        centroids = np.random.uniform(np.amin(x, axis = 0), np.amax(x, axis=0), 
                                  size = (k, x.shape[1]))
        x.to_csv("x.csv")
        return centroids 
    
def assignClusterIds(x,centroids):
    
    data_array = np.array(x)
    
    c_list = []
    
    for centroid in centroids:
        dist = []
        #print(data_array.shape)
        #print(centroids.shape)
        for data_point in data_array:

            distance = computeDistance(np.array(data_point), np.array(centroid))
            
            dist.append(distance)
        
        c_list.append(dist)

    df = pd.DataFrame(np.array(c_list)).transpose()

    clusters = df.idxmin(axis=1)  # This creates a new DataFrame with two columns: the index and the cluster IDs
    
    return clusters
    

def computeClusterRepresentatives(x, cluster_ids):
   #print(cluster_ids.values)
    # Ensure the cluster IDs are in the same order as the original data points
   x_copy = x.copy()
   x_copy['ClusterID'] = cluster_ids.values
   
   new_centroids = x_copy.groupby('ClusterID').mean()
   

   #print((new_centroids))
   centroids_array = new_centroids.values
   
   return centroids_array



def clustername(x,k,maxIter):
   centroids = initialSelection(x, k)

   for i in range(maxIter):
       C = assignClusterIds(x,centroids)
       new_centroids = computeClusterRepresentatives(x, C)
       centroids = new_centroids
       #print(centroids)

   return centroids
       
x = FileConversion()
k = 3
#initial_cluster = initialSelection(x, k)
#computeClusterRepresentatives(x,assignClusterIds(x, initialSelection(x, 3)))
clusters = clustername(x, k, 50)  

def distanceMatrix(x):
    # Compute the number of objects in the dataset
    N = len(x)
    
    # Distance matrix
    distMatrix = np.zeros((N, N))
    # Compute pairwise distances between the objects
    for i in range(N):
        for j in range (N):
            # Distance is symmetric, so compute the distances between i and j only once
            if i < j:
                distMatrix[i][j] = computeDistance(x.iloc[i], x.iloc[j])
                distMatrix[j][i] = distMatrix[i][j]
    #print(distMatrix)
    return distMatrix

distanceMatrix(x)
def silhouette_coefficient(clusters, distMatrix):
    n =len(x)
    #distances = np.zeros((n, len(clusters))

    silhouette = [0 for i in range(n)]
    a = [0 for i in range(n)]
    b = [10000000000 for i in range(n)]
    
    for (i, obj) in enumerate(x):
        for(cluster_id, cluster) in enumerate(clusters):
            clusterSize= len(cluster)
            if i in cluster:
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[i][cluster])/(clusterSize-1)
                else:
                    a[i] = 0
            else:
                tempb = np.sum(distMatrix[i][cluster])/(clusterSize)
                if tempb < b[i]: 
                    b[i] = tempb
                
    for i in range(n):
        silhouette[i] = 0 if a[i] == 0 else (b[i]-a[i])/np.max([a[i], b[i]])
    
    return silhouette
            
    
    
    
silhouette_coefficient(clusters, distanceMatrix(x))    
    
    
"""             
       distances = np.zeros((n, len(clusters)))

       for (i, point) in enumerate(x):
           for (j, center )in enumerate(clusters):
               distances[i, j] = computeDistance(point, center)
       
       cluster_assignments = np.argmin(distances, axis=1)
       print(cluster_assignments)
       silhouette = []

   silhouette_coefficient(clusters)
   """
    
    
    
    
    
    
    
    
    
    
    
    
    
    

