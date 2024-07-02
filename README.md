# som-driven-qa-rag
Self Organizing Maps (SOM) ML model can be used to conduct semantic search to populate context required for Retrieval Augmented Generation (RAG) LLM models. This repo contains an example to demonstrate the SOM capability.

# UMAP-TSNE-SOM Comparision
This branch has similar classes to the original SOM, using either TSNE or UMAP for dimetionality reduction. 

SOM Training Workflow (fully pytorch):
1. Initialize a 2d lattice of fixed size, and an associated weights vector with the same shape. Weight vector nodes have the same dimensions as the input data, and are initialized to random weights. 
2. Data is passed to training function
3. For some number of epochs, we iterate through every vector in the input data, incrementally adjusting the weights of the nodes. For each input vector, a best matching node is found in the lattice (most similar weight), that lattice node and the ones near it in the low dimensional lattice then have their weights adjusted to be "pulled" closer to the input vector. Repeat for all vectors for a number of epochs.
4. The result is a 2d lattice with associated weights corresponding to centroids in the high dimentional space. Lattice nodes closer together in the 2d lattice correspond to centroids closer together in the high-dimentional space. 

UMAP/TSNE  Training Workflow (numpy):
1. Init a TSNE/UMAP instance with the relevant parameters
2. Data passed to training function
3. Convert data to nump array
4. fit_transform the input data to the TSNE/UMAP instance, result is a numpy array of flattened data
5. Run k-means clustering on the flattened data to find clusters.
6. Use clusterings to find indexes of vectors that are in clusters together
7. Use those indexes to group the high dimentional vectors together in clusters, then find the high dimetional means of the clusters
8. High dimentional means are to be used as the equivalent of the lattice node weights in SOM

