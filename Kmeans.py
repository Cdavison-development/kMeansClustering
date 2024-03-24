#specify the number k of clusters to assign
#Randomly initialise k centroids
#repeat
  #expectation: assign each point to its closest centroid
    #maximisation: compute the new centroid (mean) of each cluster
#until the centroid positions do not change



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
        return numeric_data
        
def computeDistance(a,b):
        dist = np.linalg.norm(a-b)
        return dist

def initialSelection(x,k):
    
        np.random.seed(1500)
        centroids = np.random.uniform(np.amin(x, axis = 0), np.amax(x, axis=0), 
                                  size = (k, x.shape[1]))
        #print(centroids)
        return centroids 
    
#def computeClusterRepresentatives():
    
def assignClusterIds():
    data_points = FileConversion()
    centroids = initialSelection(FileConversion(), 3)
    data_array = np.array(data_points)
    c_list = []
    #data_list = data_points.values.tolist()
    centroid_list = centroids.tolist()
    for centroid in centroid_list:
        dist = []
        for data_point in data_array:
            distance = computeDistance(np.array(data_point), np.array(centroid))
            dist.append(distance)
        
        c_list.append(dist)
            
    df = pd.DataFrame(np.array(c_list)).transpose()
    
    clusters = df.idxmin(axis=1)
    return clusters
    

def computeClusterRepresentatives(x, cluster_ids):
    # Ensure the cluster IDs are in the same order as the original data points
   x['ClusterID'] = cluster_ids.values

   # Group the data points by their assigned cluster and calculate the mean
   new_centroids = x.groupby('ClusterID').mean()

   print(new_centroids)
   return new_centroids
#def clustername(x,k):
    

            
computeClusterRepresentatives(FileConversion(),assignClusterIds())
    
 
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




