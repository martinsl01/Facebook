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

# CONVERT TO FEATURE EXTRACTION MODEL

class ImageClassifier(torch.nn.Module):

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

    def forward(self,X):
       return self.combo(X)

def train(model, train_dataloader, val_dataloader, test_dataloader, epochs=50):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    writer = SummaryWriter()    
    batch_idx = 0
    print(device)
    model.to(device)

    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H_%M_%S")      
    folder_path = f'~/FMRRS/final_model/{timestamp}/'    
    os.makedirs(folder_path, exist_ok = True) 

    for epoch in range(epochs):
        for i, (features,labels, image_id) in enumerate(train_dataloader):
            features = features.to(device)
            print(features.shape)
            labels = labels.to(device)
            prediction = model(features)
            train_loss = F.cross_entropy(prediction,labels)
            train_loss.backward()
            optimizer.step()
            train_accuracy = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            print(f"Epoch:{epoch}, Batch number:{i}, Training: Loss: {train_loss.item()}, Training Accuracy: {np.mean(train_accuracy)}")
            optimizer.zero_grad()   
            writer.add_scalar('loss', train_loss.item(),batch_idx)
            batch_idx += 1

        '''Evaluation of model on validation dataset'''

        val_loss, val_accuracy = evaluate(model, val_dataloader)
        writer.add_scalar("Loss/Val", val_loss, batch_idx)

        '''Saving model'''
        
        weights =  f'~/FMRRS/final_model/{timestamp}/weights'  
        os.makedirs(weights,exist_ok=True)
        torch.save(model.state_dict(), os.path.join(weights, f'image_model{epoch}.pt'))
        
        print('train loss type: ', type(train_loss), train_loss)
        print('val loss type: ', type(val_loss), val_loss)
        print('train accuracy: ', type(train_accuracy), train_accuracy)
        print('val acc: ', type(val_accuracy), val_accuracy)
        metrics = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        metrics[f'train_loss'].append(train_loss.tolist())
        metrics[f'val_loss'].append(val_loss.tolist())
        metrics[f'train_accuracy'].append(train_accuracy)
        metrics[f'val_accuracy'].append(val_accuracy.tolist())

        metrics_pth =  f'~/FMRRS/final_model/{timestamp}/metrics'  
        os.makedirs(metrics_pth, exist_ok=True)
        with open(os.path.join(metrics_pth, f'image_metrics{epoch}.json'), 'w') as file:
            json.dump(metrics, file)

        print(metrics['train_loss'], metrics['val_loss'], metrics['train_accuracy'], metrics['val_accuracy'])
        print(metrics.values())

        '''Evaluation of final test set performance'''

        test_loss, test_accuracy = evaluate(model, test_dataloader)
        model.test_loss = test_loss
        print('test loss: ', test_loss, 'test accuracy: ', test_accuracy)
 

'''Function to evaluate model on unseen datasets'''
def evaluate(model, dataloader):
    losses = []
    accuracy = []
    for batch in dataloader:
        features, labels, Image_id = batch
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
        acc = torch.sum(torch.argmax(prediction,dim=1) == labels).item()/len(labels)
        accuracy.append(acc)
    avg_accuracy = np.mean(accuracy)
    avg_loss = np.mean(losses)
    print('Average Loss',avg_loss, "| Average Accuracy", avg_accuracy)
    return avg_loss, avg_accuracy

dataset = ProductsDataset()
# Split the dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=16)
model = ImageClassifier()
train(model, train_dataloader, val_dataloader, test_dataloader, epochs=50)   