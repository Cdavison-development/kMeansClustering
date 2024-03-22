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
    
        np.random.seed(10)
        centroids = np.random.uniform(np.amin(x, axis = 0), np.amax(x, axis=0), 
                                  size = (k, x.shape[1]))
        #print(centroids)
        return centroids 
    

def clustername(x,k):
    
    data_points = FileConversion()
    data_array = np.array(data_points)
    centroids = initialSelection(FileConversion(), 3)
    
    # Stores the centroid values, as they are updated, initialiased to zeros
    data_array_old = np.zeros(data_array.shape)
# Stores the centroid nearest to the point
    clusters = np.zeros(len(x))
# Stores the error at this stage, iteration runs till error becomes zero
    error = computeDistance(data_array,data_array_old,None)
    while error!=0:
        # Assigning each point to its nearest cluster
        for i in range(len(data_array)):
            distances = computeDistance(x[i],centroids)
            cluster = np.argmin(distances)
            clusters[i] = cluster
            # Old centroid values stored in c_old 
            data_array_old = deepcopy(centroids)
            # Finding the new mean of each cluster
            for i in range(k):
                points = [x[j] for j in range(len(x)) if clusters[j] == i]
                centroids[i] = np.mean(points, axis=0)
            error = computeDistance(centroids, data_array_old, None)
    
    print(centroids)
    print(clusters)
    print(error)
    
    
    
    
    
    
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




