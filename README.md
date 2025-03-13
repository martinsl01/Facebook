  # Facebook Marketplace Recommendation Ranking System

## CONTENTS

- [Introduction](#introduction)
- [Environment](#environment)
- [Sourcing Data](#sourcingdata)
- [Data Preparation](#datapreparation)
- [Models](#themodels)
- [Faiss](#faiss)
- [Deploying Model](#deployingmodel)

  ## Introduction

The task is about building a search index that is operationally adjacent to ‘Facebook Marketplace’, a platform for buying and selling goods on Facebook. This readme file documents the implementation of the system behind the marketplace, which uses Artificial Intelligence to recommend the most relevant listings based on a personalised search query. 

![Screenshot 2025-02-15 163841](https://github.com/user-attachments/assets/dadac9a6-d52b-4a3f-8e67-bbef60a51e89)

There are four major steps which was required to complete the project, first was to set up the right environment, both locally and remotely. The second was to retrieve the data that would be used for training and finetuning the models later on. Third was to load a pretrained model and adapt it to the task and then alter it to extract the features of the many images in the dataset. The extracted features were then used to group together images that had things in common with whatever search query is entered. 

Finally, after the search index was created, a docker image was built in order to containerise the model, ensuring it can be functional on other computers with different operating system and/or environment. The model was deployed using Fastapi, and it worked well to satisfy the queries entered into it. The evidence is provided through the screenshots, and the model could also be run by anyone as long as the required tools i.e. libraries and applications are satisfied. 

## Environment

**Local:**
In order to ensure things go as smoothly as possible, it’s always important to begin by setting up a virtual environment locally. With this, every dependency such as libraries and packages required for the task can be in one place and prevent conflicts between dependencies, a common one being versions of certain dependencies due to the constant updates of open-source libraries and the like. 

**Remote:**
A github repository was created to track changes to the code which was advantageous on multiple fronts. It gives anyone who receives the cloned link the ability to view the code from their end and if they choose to, test my work by running the whole thing. Also, it is a fail-safe in case something goes drastically goes wrong with the code in the local machine, an example would be having files permanently deleted. In that event then I can ‘git pull’ or ‘merge’ to recover lost or damaged files. 

## Sourcing Data

The dataset and images required for the task was in an EC2 instance on AWS, one of the major cloud software that allows for the depositing and storing of information, also with the facility to write code on it as well. I accessed the account and downloaded a private key from the S3 bucket which I used to SSH into the EC2 instance. In there, were two ‘.csv’ files and a folder, the former were the dataset containing information about the listings such as price, location and category whilst the latter were the images corresponding to the listings in the dataset. 

## Data Preparation

**Cleaning Tabular Dataset**

This part of the task is all about making the data usable so it can be fed into the models to be created so as to ensure their efficacy. Both the tabular and image dataset required cleaning before they could be used. 

The tabular dataset contained lots of null values which I dutifully removed using the ‘.dropna()’ function. The prices column was converted into a numerical format; removed all pound signs and commas to make the conversion functional and universal. 

**Extracting Labels**

The next thing was to extract labels for classification, and this was done by merging the two ‘.csv’ files together. The two files were ‘products.csv’ which had various information about the product such as price and location, whilst the ‘image.csv’ file contained the ‘image_id’ and ‘product_id’, and given that both dataset possessed the ‘product_id’ column, that was used to merge the two seemingly, making the dataset one. The ‘image_id’ and ‘categories’ were now in the same table, allowing for the accurate classification of each product. It’s worth noting that an encoder was created to make the classification possible, with each category being assigned a number between 0-12 (shown below). A [decoder](decoder.json) was also created to provide the opposite mapping. 

![Screenshot 2025-02-15 182006](https://github.com/user-attachments/assets/0d3551d5-6ed1-4eac-933f-cdf35e871ecc)

**Cleaning Image Dataset**

As for the images, they all need to be consistent in terms of size and the number of channels. The standerdisation is required to prevent the results of the training, feature extraction and ultimately the search index from being compromised. I created the [clean_images.py](clean_images.py) script and wrote code to clean the image dataset and Create a pipeline that applies the necessary cleaning to the image dataset. After the images passed through the pipeline, they were all adjusted to having three (RGB) channels and 512 pixels as shown in the screenshot - 'property' section of one of the cleaned images below. 

<img width="300" alt="Screenshot 2025-02-15 185316" src="https://github.com/user-attachments/assets/1b90c20e-0448-4cae-9eb8-4f78fba47cef" />


**Pytorch Dataset**

The image dataset to be fed into the classification model is fed through a [Pytorch Dataset](pytorch_dataset.py), which through the __getitem__() method, allocates appropriate labels to each image. The images were transformed using the ‘transforms’ from the ‘torchvision’ that applied necessary changes such as ‘RandomHorizontalFlip’, conversion of features to ‘Tensor’ and ‘Normalize’. The dataset was coded to return ‘features’, ‘labels’ and the ‘image_id’ which were essentials for the tasks ahead. 

## Models

**Pretrained Model**

When it came to the creation of the models, I took advantage of the CNN architecture that had already been created and trained by others, known as resnet50 on Pytorch. Other than saving time, another major benefit of this is that the intensive training that is often required to get very effective and accurate models has been done in much more capable systems than the one I’m working with, so I don’t have to push my computer to the limits. Therefore, using transfer learning, I fine-tuned the 'Pretrained Model' to classify the images from the Pytorch Dataset as accurately as possible. To do so, I replaced the final linear layer of the model with another linear layer whose output size is the same as the number of categories – 13. 
Following on from this I defined a function called train, used to polish the model, ensuring it’s efficacy. It takes in a model as its first positional argument and also in a keyword argument, epochs for the number of times it will be trained for. Inside the function, the batch of the dataset are in a loop, updating the model's parameters each time. The loss is printed after every prediction to get an idea of how well the training is going. The dataset was split up into Training, validation and Test sets to monitor the training and tensorboard was launched (as shown in the image below) to measure the performance over time by plotting the training loss. Multiple epochs were run in order to enhance the model, and to prevent overfitting or underfitting. 

![Screenshot 2025-02-15 220817](https://github.com/user-attachments/assets/c7df0645-f04a-4da7-b16e-413ea3b4842e)


The training loop was coded to save the weights of the model at the end of every epoch (see ~/FMRRS/model_eval). The saving was coded to include the timestamp of when the model was used in the name of the folder and within that model folder, it creates a folder called weights where the weights of the model was saved with a filename that shows what epoch each saved weights corresponded to. 

**Feature Extraction Model**

The heart of the entire project is to extract feature vectors from images, so they can be grouped with other images with a similar feature vector. In order to do this the features of the images need to be extracted, and this is done in a straightforward way – converting the classification model into a feature extraction model. The process is to firstly remove the last few fully connected (fc) layers from the model, then design the last layer of this feature extraction model so that it should have 1000 Neurons (in this case). A new folder was then created to save the training of the feature extraction model with it’s 1000 Neurons. In order to get embeddings from the feature extraction model, the last layer was redesigned for that purpose, shown in the screenshot below. The embeddings were saved as a dictionary in json, ready to be used for the search index. 

<img width="427" alt="Screenshot 2025-02-15 220309" src="https://github.com/user-attachments/assets/2cad6f34-13ac-48b5-8e1c-8e290fe3d533" />


## Faiss

The first step in building the Faiss Search index was to load the image_embeddings dictionary from the previous part of the project and fit it into the Index. Faiss needs two things for a vector search, the first is an index for the vector data and second is the vector data. The image_ids (keys in the loaded dict) served as the index whilst the image_embeddings (values in the loaded dict) were the feature_vectors. After this, a vector search was performed, with the idea to extract features of any given image and search for similar images using the Faiss Search Index. [See here](faiss_script.py)

A screenshot of a query with k (nearest neighbour) set at 5 and the respective distances is below. 

![Screenshot 2025-02-15 224246](https://github.com/user-attachments/assets/f766accf-4f94-4870-8866-c9f6b9ff49e2)

## Deploying Model

Two API enpoints were set up with relevant methods from each model to run predictions on the relevant request body (images and other params) and then return the predictions/search result indexes. I initialised the models I wanted to use by defining the same architecture used to train each model. Then, I initialised an instance of the model and loaded the weights I obtained from training and exporting a .pt file, the one with the most accurate parameters. 

After this I moved every file I needed into a folder called [app](app), then built a docker image to containerise my work, ensuring it can function on other systems/environments. I checked that the API was working by visiting [ http://localhost:8080/docs] in my browser. The image below is a screenshot of where the image file is uploaded as the search query. 

![Screenshot 2025-02-27 233012](https://github.com/user-attachments/assets/6ab885a9-850a-483f-b7a7-9ea733ae34b7)

The screenshot below is the responde body which contains useful information - category of images and distance from search query. Finally I pushed the image to Docker Hub.

![Screenshot 2025-02-27 233040](https://github.com/user-attachments/assets/608d2058-f73b-48d9-bfad-1a8fbc46bff8)








