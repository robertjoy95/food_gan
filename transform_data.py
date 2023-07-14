import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from constants import *

def get_test_train_data():
    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        transforms.CenterCrop(image_size),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = datasets.ImageFolder(root="data/food-101/subset", # target folder of images
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)
    
    
    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True) # shuffle the data?

    
    return train_dataloader

    

if __name__ == "__main__":
    dataloader = get_test_train_data()
