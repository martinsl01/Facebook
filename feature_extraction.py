import torch 
from pytorch_dataset import ProductsDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
import json
from image_processor import process_image
from PIL import Image

# CONVERT TO FEATURE EXTRACTION MODEL

class FeatureExtractor(torch.nn.Module):

    '''This loads the Resnet50 pre-trained model from torch hub and alters the final layer.'''
    def __init__(self):
        super().__init__()
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

    def forward(self,X):
       return self.extraction(X)
    
def image_embedding(image):
    with torch.no_grad():
        feature_extractor = FeatureExtractor()
        embeddings_dict = {}
        
        for features, labels, image_id in image:

            feat = feature_extractor.forward(features)
            flat_1 = feat.detach().numpy()
            flat_2 = flat_1.flatten() 
            embeddings = flat_2.tolist() # Converts the tensor into a list 
            embeddings_dict[str(image_id)] = embeddings  
            
        features_pth =  '~/FMRRS/final_model/features'  
        os.makedirs(features_pth, exist_ok=True)
        with open(os.path.join(features_pth, f'image_embeddings.json'), 'w') as file:
            json.dump(embeddings_dict, file)                                                              

    
dataset = ProductsDataset()
# Split the dataset
train_size = int(1.0 * len(dataset))
val_size = int(0.0 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=1)
image_embedding(train_dataloader)