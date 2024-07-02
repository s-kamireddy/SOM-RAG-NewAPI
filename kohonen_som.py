# coding: utf-8
'''
------------------------------------------------------------------------------
   Copyright 2024 Murali Kashaboina

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
------------------------------------------------------------------------------
'''

import warnings

warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import sklearn

from tqdm.autonotebook import tqdm

from typing import List

class KohonenSOM():
    """
    The code is developed based on the following article:
    http://www.ai-junkie.com/ann/som/som1.html
    
    The vector and matrix operations are developed using PyTorch Tensors.
    """
    def __init__( 
                    self,
                    input_dimensions : int, 
                    som_lattice_height : int = 20,
                    som_lattice_width : int = 20,
                    learning_rate : float = 0.3,
                    neighborhood_radius : float = None,
                    device : str = None 
                ):

        self.input_dimensions = int( input_dimensions ) #dimensions of each vector in the input

        self.som_lattice_height = int( som_lattice_height ) 

        self.som_lattice_width = int( som_lattice_width )

        if learning_rate == None:
            self.learning_rate = 0.3
        else:
            self.learning_rate = float( learning_rate )

        if neighborhood_radius == None: #neighborhood_radius helps to decide how many neighbors get pulled along with the best matching node in training
            self.neighborhood_radius = max( self.som_lattice_height, self.som_lattice_width ) / 2.0
        else:
            self.neighborhood_radius = float( neighborhood_radius )

        if device == None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = torch.device( device )
        
        def dist_eval( data_points, weights ): #dist eval function to be used in training
            distances = torch.cdist( data_points, weights,  p=2 )
            #print(f"data_points shape:{data_points.shape }  weight shape {weights.shape}")
            return distances
            
        self.dist_evaluator = dist_eval
        
        self.total_lattice_nodes = self.som_lattice_height * self.som_lattice_width
        
        self.lattice_node_weights = torch.randn( self.total_lattice_nodes, self.input_dimensions, device=self.device ) #randomly init corresponding array of high-dim weights
        
        lattice_coordinates = torch.tensor( [ [[i,j] for j in range(self.som_lattice_width)] for i in range( self.som_lattice_height ) ], dtype=torch.int ) #init a 2-d lattice

        self.lattice_coordinates = lattice_coordinates.view( self.total_lattice_nodes, 2 )
        
        self.trained = False
        
        #self.dist_evaluator = nn.PairwiseDistance(p=2)

    def train( self, data_points : torch.Tensor, train_epochs : int = 100 ):
        if self.trained:
            print( "WARNING: Model is already trained. Ignoring the request..." )
            return
        
        train_epochs = int( train_epochs ) #how many times we iterate over all the nodes
        
        total_dpoints = data_points.shape[0] #number of input vectors
        
        data_points = data_points.to( self.device )
        
        for epoch in tqdm( range( train_epochs ), desc="Kohonen's SOM Train Epochs" ):
            decay_factor = 1.0 - (epoch/train_epochs) #decay
            
            #learning rate is alpha in the paper, ie less of an impact training in later epochs
            adjusted_lr = self.learning_rate * decay_factor 
        
            #sigma in the paper
            adjusted_lattice_node_radius = self.neighborhood_radius * decay_factor 
            
            #sigma square in the paper
            squared_adjusted_lattice_node_radius = adjusted_lattice_node_radius**2

             #evaluate distances between all lattice node weights and all input vectors
            distances = self.dist_evaluator( data_points, self.lattice_node_weights )
            
            best_matching_units = torch.argmin( distances, dim=1 ) #find the indexes of the best matching lattice weight node to each input vector
            
            for i in range( total_dpoints ):
                data_point = data_points[i] #pick one input vector
                
                bmu_index = best_matching_units[i].item() #get BMU index
                
                bmu_coordinates = self.lattice_coordinates[ bmu_index ] #get the coordinates in the 2d-lattice
                
                #squared distances of the lattice nodes from the bmu :: dist^2 from equation 6 shown in the paper
                squared_lattice_node_radii_from_bmu = torch.sum( torch.pow( self.lattice_coordinates.float() - bmu_coordinates.float(), 2), dim=1)
                
                squared_lattice_node_radii_from_bmu = squared_lattice_node_radii_from_bmu.to( self.device )
                
                #adjust function phi in the paper
                lattice_node_weight_adj_factors = torch.exp( -0.5 * squared_lattice_node_radii_from_bmu / squared_adjusted_lattice_node_radius ) #nodes closer to bmu are adjusted more
                
                lattice_node_weight_adj_factors = lattice_node_weight_adj_factors.to( self.device )
                
                final_lattice_node_weight_adj_factors = adjusted_lr * lattice_node_weight_adj_factors #apply lr to adjustments
                
                final_lattice_node_weight_adj_factors = final_lattice_node_weight_adj_factors.view( self.total_lattice_nodes, 1 )
             
                final_lattice_node_weight_adj_factors = final_lattice_node_weight_adj_factors.to( self.device )
                
                lattice_node_weight_adjustments = torch.mul( final_lattice_node_weight_adj_factors, (data_point - self.lattice_node_weights) )
                
                self.lattice_node_weights = self.lattice_node_weights + lattice_node_weight_adjustments #adjust the lattice node weights
                
                self.lattice_node_weights = self.lattice_node_weights.to( self.device )
        print(f"data points dimensions {data_points.shape}  lattice dimensions {self.lattice_node_weights.shape}")
        self.trained = True

    def find_best_matching_unit( self, data_points : torch.Tensor ) -> List[ List[ int ] ] : #takes input vectors and returns best matching unit from the lattice node weights
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
            
        distances = self.dist_evaluator( data_points, self.lattice_node_weights )
        
        best_matching_unit_indexes = torch.argmin( distances, dim=1 )
        
        best_matching_units = [ self.lattice_coordinates[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
        
        #print("BMU shape:")
        #print(best_matching_units.shape())
        
        return best_matching_units
    
    def find_topk_best_matching_units( self, data_points : torch.Tensor, topk : int = 1 ) -> List[ List[ int ] ] : #takes multiple input vectors and returns topk best matching unit for all
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
        
        topk = int( topk )
        
        distances = self.dist_evaluator( data_points, self.lattice_node_weights )
        
        topk_best_matching_unit_indexes = torch.topk( distances, topk, dim=1, largest=False ).indices
        
        topk_best_matching_units = []
        
        for i in range( data_points.shape[0] ):
            best_matching_unit_indexes = topk_best_matching_unit_indexes[i]
            
            best_matching_units = [ self.lattice_coordinates[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
            
            topk_best_matching_units.append( best_matching_units )
        #print(f"topk matching units shape: {topk_best_matching_units.shape()}")
        return topk_best_matching_units
