import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

class ProductsDataset(Dataset):

    def __init__(self, annotations_file = 'C:/Users/marti/FMRRS/training_data.csv', 
                 img_dir = 'C:/Users/marti/FMRRS/cleaned_images/', transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([   
        transforms.Resize(224),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.loc[index, 'image_id']}.jpg")
        features = Image.open(img_path)
        label = self.img_labels.loc[index, 'labels']
        img_id = img_path.split("/")[-1]
        image_id = img_id.split(".")[0]
        if self.transform:
            features = self.transform(features)
        else:
    # Optionally convert to tensor directly
            features = transforms.ToTensor()(features)
        return features, label, image_id


if __name__ == '__main__':

    dataset = ProductsDataset()
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  

# print(dataset[200])






