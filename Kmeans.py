
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
            #print(f"File converted and saved as '{output_file}'")

        df = pd.read_csv('converted_file.csv')
        numeric_data = df.iloc[:, 1:]
        numeric_data.to_csv("numeric.csv")
        return numeric_data
        

def computeDistance(a,b):
        dist = np.linalg.norm(a-b)
        return dist

def initialSelection(x,k):
    
        np.random.seed(314159)
        centroids = np.random.uniform(np.amin(x, axis = 0), np.amax(x, axis=0), 
                                  size = (k, x.shape[1]))
        #print(centroids)
        return centroids 
    
#def computeClusterRepresentatives():
    
def assignClusterIds(x,J):
    #data_points = x
    J = initialSelection(x, 3)
    #print(centroids)
    data_array = np.array(x)
    print(len(x))
    c_list = []
    
    for centroid in J:
        dist = []
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
   x_copy = x
   x_copy['ClusterID'] = cluster_ids.values
  
   new_centroids = x_copy.groupby('ClusterID').mean()
   

   print((new_centroids))
   centroids_array = new_centroids.values
   return centroids_array



def clustername(x,k,maxIter):
   centroids = initialSelection(x, k)

   for i in range(maxIter):
       C = assignClusterIds(x,centroids)
       new_centroids = computeClusterRepresentatives(x, C)
       #print(new_centroids)
       centroids = new_centroids
   #print(centroids)
   return centroids
       
x = FileConversion()

#computeClusterRepresentatives(x,assignClusterIds(x, initialSelection(x, 3)))
 # Call FileConversion once and use 'data'
clustername(x, 9, 9)  

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cluster_ids, cmap='viridis', marker='o')
plt.title("Cluster Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster ID')
plt.show()
#clustername(FileConversion(), 8, 200)
            
#computeClusterRepresentatives(FileConversion(),assignClusterIds())
    
 
#clustername(FileConversion(),100)    
    
    
    
    
    
""""
    data_points = FileConversion()
    data_array = np.array(data_points)
    centroids = initialSelection(FileConversion(), 3)
    
    print(type(centroids))
    for _ in range(1,k):
       dist = []
       for i in range(len(data_array)):
           point = data_array[i]
           min_dist_with_cen = computeDistance(point, centroids[0])
           if len(centroids) > 1:
               for j in range(1,len(centroids)):
                   dist_wrt_centroid_j = computeDistance(point, centroids[j]) 
                   min_dist_with_cen = min(min_dist_with_cen, dist_wrt_centroid_j)
                   
           dist.append(min_dist_with_cen)
       
       dist = np.array(dist)
       next_centroid = data_array[np.argmax(dist)]
       
       np.append(centroids, next_centroid)
       
       print(centroids)
    return centroids
        
        
        
clustername(FileConversion(),100)


        
        
        
def initialisation(x,k):

def computeClusterRepresentatives(C):

def assignClusterIds(x,k,Y):
    
def kMeans(x,k,maxIter):
    initialisation(x,k)
    for i in range(1, maxIter):
        C = assignClusterIds(x,k,Y)
        Y = computeClusterRepresentatives(C)


    return clusters
    
def ComputeSilhouettee():
    
def plot_silhouettee():


"""""
