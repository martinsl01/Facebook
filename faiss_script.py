import torch
import numpy as np
import json
import pickle
import faiss
import os


'''Loading the image embeddings'''

with open("C:/Users/marti/FMRRS/image_embeddings.json", "r") as f:
    embeddings = json.load(f)

'''Converting dictionary to numpy array'''

image_ids = list(embeddings.keys())

tensor_list = [torch.tensor(values) for values in embeddings.values()]
feature_vectors = torch.stack(tensor_list).numpy()
print(feature_vectors.shape)

'''Faiss requires two things and one is an index and the other is a vector data.
image_ids will be the index whilst the image embeddings will be the vector data'''

# Faiss

dimension = feature_vectors.shape[1]
print(dimension)
index = faiss.IndexFlatL2(dimension)

# Feature Vector
index.add(feature_vectors)
'''Faiss saved for later use'''

faiss.write_index(index, 'faiss_index.pkl')
'''The next stage is to perform search using the Faiss Search Index, not just any image but those 
which are similar based on their feature_vector'''

def similar_images_search(query, top_k=10):
    query = query.reshape(1,-1) # Reshape for Faiss#
    distances, indices = index.search(query, top_k) # Search Faiss index

    # Getting similar images
    similar_images = [image_ids[idx] for idx in indices[0]]
    print(similar_images, distances[0])

query_1 = np.random.rand(1, dimension).astype("float32")
# print(query_1)
k = 5

similar_images_search(query_1, k)


