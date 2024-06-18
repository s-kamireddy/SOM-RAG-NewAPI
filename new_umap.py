import warnings

warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from typing import List

class UMAP_Util():
    """
    The code is developed based on the following article:
    http://www.ai-junkie.com/ann/som/som1.html
    
    The vector and matrix operations are developed using PyTorch Tensors.
    """
    def __init__( 
                    self,
                    input_dimensions : int, 
                    n_neighbors : int = 15,
                    min_dist : float = 0.2,
                    n_components : int = 2,
                    k_clusters : int = 2,
                    metric : str = 'euclidean',
                    device : str = None  
                ):

        if device == None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = torch.device( device )

        def dist_eval( data_points, weights ):
            distances = torch.cdist( data_points, weights,  p=2 )
            #print(f"data_points shape:{data_points.shape }  weight shape {weights.shape}")
            return distances

        self.dist_evaluator = dist_eval

        self.reducer = UMAP(n_neighbors = n_neighbors, min_dist = min_dist, n_components = n_components, metric = metric,)

        self.kmeans = KMeans(k_clusters)

        self.trained = False

        #self.embedding = None
        
        self.centroids = torch.zeros(k_clusters, input_dimensions, device = self.device) #high dim centroids 
        
        self.cluster_centers = None #low dim centroids

        #self.dist_evaluator = nn.PairwiseDistance(p=2)




    def train( self, data_points : torch.Tensor):
        if self.trained:
            print( "WARNING: Model is already trained. Ignoring the request..." )
            return

        data_points = data_points.cpu().numpy() #convert data to np array to run umap

        self.reducer.fit(data_points) #fit model

        embedding = self.reducer.transform(data_points)

        

        self.kmeans.fit(embedding)#kmeans on flattened data

        self.cluster_centers = torch.tensor(self.kmeans.cluster_centers_)
        
        plt.scatter( embedding[:, 0],
                    embedding[:, 1], c = self.kmeans.predict(embedding))
        
        plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], c='black', s =  200,  alpha=0.5);
        
        #calculate high dimentional means of those clusters
        dict = {}
        for i in range(0, data_points.shape[0]):
            if not self.kmeans.labels_[i] in dict.keys():#group in dict?
                dict[self.kmeans.labels_[i]] = []
            dict[self.kmeans.labels_[i]].append([data_points[i]])# add the data vector



        for i in range(0, self.kmeans.cluster_centers_.shape[0]):
            temp = np.concatenate(dict[i], axis = 0)
            tensor_mean = torch.tensor(np.mean(temp, axis = 0))
            self.centroids[i] = tensor_mean #add high dim mean to centroids
        
        data_points = torch.tensor(data_points).to(self.device)

        self.trained = True

        
    def find_best_matching_unit( self, data_points : torch.Tensor ) -> List[ List[ int ] ] :
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
        
        #flatten data_points to compare to our 2d centroids
        distances = self.dist_evaluator( data_points, self.centroids ) #calculate distances
        
        best_matching_unit_indexes = torch.argmin( distances, dim=1 )
        
        best_matching_units = [ self.cluster_centers[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
        
        #print("BMU shape:")
        #print(best_matching_units.shape())
        
        return best_matching_units

    def find_topk_best_matching_units( self, data_points : torch.Tensor, topk : int = 1 ) -> List[ List[ int ] ] :
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
            
        #flatten data_points to compare to our 2d centroids
        
        topk = int( topk )
        
        distances = self.dist_evaluator( data_points, self.centroids)
        
        topk_best_matching_unit_indexes = torch.topk( distances, topk, dim=1, largest=False ).indices
        
        topk_best_matching_units = []
        
        for i in range( data_points.shape[0] ):
            best_matching_unit_indexes = topk_best_matching_unit_indexes[i]
            
            best_matching_units = [ self.cluster_centers[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
            
            topk_best_matching_units.append( best_matching_units )
        #print(f"topk matching units shape: {topk_best_matching_units.shape()}")
        return topk_best_matching_units


