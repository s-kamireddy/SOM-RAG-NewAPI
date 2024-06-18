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

from tqdm.autonotebook import tqdm

import torch

from tsne import TSNE_Util

from typing import List


class TSNEBasedVectorIndexer():
    def __init__( 
                    self,
                    input_dimensions : int,
                    n_components : int = 2,
                    perplexity : int = 30,
                    epochs : int = 1000,
                    k_clusters : int = 2,
                    metric : str = "euclidean", 
                    topk_bmu_for_indexing : int = 1,
                    device : str = None 
                ):
        self.topk_bmu_for_indexing = int( topk_bmu_for_indexing )
        
        self.tsne_util = TSNE_Util(
                                input_dimensions = input_dimensions, 
                                n_components = n_components,
                                perplexity = perplexity,
                                epochs = epochs, 
                                metric = metric,
                                k_clusters = k_clusters,
                                device = device
                             )
        
        self.generated_indexes = False
        
        self.tsne_node_idx_map = {}
    
    def train_n_gen_indexes( 
                                self, input_vectors : torch.Tensor, 
                           ):
        if self.generated_indexes:
            print( "WARNING: Indexes were already generated. Ignoring the request..." )
            return
        
        
        self.tsne_util.train( input_vectors )
        
        topk_bmu_indexes = self.tsne_util.find_topk_best_matching_units( input_vectors, topk = self.topk_bmu_for_indexing ) #get the lattice nodes closest to each vector
        
        for idx in tqdm( range( len( topk_bmu_indexes ) ), desc="SOM-Based Indexed Vectors"  ):
            bmu_indexes = topk_bmu_indexes[ idx ]
            
            for bmu_index in bmu_indexes:
                bmu_index_key = tuple( bmu_index )
            
                idx_set = self.tsne_node_idx_map.get( bmu_index_key, set() )
            
                idx_set.add( idx )
                
                self.tsne_node_idx_map[ bmu_index_key ] = idx_set
        
        self.generated_indexes = True
    
    def find_nearest_indexes( self, input_vectors : torch.Tensor ) -> List[ List[ int ]  ]:
        topk_bmu_indexes = self.tsne_util.find_topk_best_matching_units( input_vectors, topk = self.topk_bmu_for_indexing )
        
        nearest_indexes = []
        
        for idx in range( len( topk_bmu_indexes ) ):
            nearest_idx = set()
            
            bmu_indexes = topk_bmu_indexes[ idx ]
            
            for bmu_index in bmu_indexes:
                bmu_index_key = tuple( bmu_index )
                
                neighbor_idx_set = self.tsne_node_idx_map.get( bmu_index_key, set() )
                
                nearest_idx = nearest_idx.union( neighbor_idx_set )
            
            nearest_idx = list( nearest_idx )
            
            nearest_idx.sort()
            
            nearest_indexes.append( nearest_idx )

        return nearest_indexes