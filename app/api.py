import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import json
import torch
import pandas as pd
import torch.nn as nn
from pydantic import BaseModel
from torchvision import transforms
import numpy as np
import faiss

class FeatureExtractor(torch.nn.Module):

    '''This loads the Resnet50 pre-trained model from torch hub and alters the final layer.'''
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        for param in self.resnet50.parameters(): 
            param.requires_grad = False
        self.added_layer = torch.nn.Sequential( 
            torch.nn.ReLU(),
            torch.nn.Linear(1000,1000))
        self.combo = torch.nn.Sequential( 
            self.resnet50, self.added_layer)
        self.extraction = nn.Sequential(*list(self.combo.children())[:-1])
        
        with open('decoder.json', 'r') as f:
            decoder = json.load(f)

        self.decoder = decoder

    def forward(self,X):
       return self.extraction(X)

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:

    feature_ext = FeatureExtractor()
    weight_dict = torch.load("image_model8.pt")
    feature_ext.load_state_dict(weight_dict, strict=False)

    pass    
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    index = faiss.read_index('faiss_index.pkl')
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

# transforms:

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

'''The endpoint below is to process the image, extract it's features and use it to find other images
with similarities'''

with open('image_embeddings.json', 'r') as file:
    data = json.load(file)

image_embeddings = data

@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    tran_image = transform(pil_image)
    processed = tran_image.unsqueeze(0)
    extraction = feature_ext(processed)
    query = extraction.tolist()[0]
    distances, indices = index.search(np.array(query, dtype=np.float32).reshape(1,-1), 5) 
    image_ids = list(image_embeddings.keys())  # Convert keys to a list
    images_list = [image_ids[idx] for idx in indices[0]]
    similar_images = [s.replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace(",", "") for s in images_list]
    distances_list = distances.tolist()[0]
    indices_list = indices.tolist()[0]

    dataset = pd.read_csv('training_data.csv')
    data = dataset.loc[dataset['image_id'].isin(similar_images)]
    categories = data['category'].tolist()
    results = {'similar_image_id': similar_images, 'distance': distances_list, 'nearest_neighbour': indices_list, 'category': categories}

    return results
        
if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080) 