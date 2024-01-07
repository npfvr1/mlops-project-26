import os
import torch
import torchvision
from torchvision import datasets, transforms


if __name__ == '__main__':

    # Adjusted transform by resizing to 256 (was before 224), and centercrop to 224, 
    # which is a standard approach for preparing images for models like ResNet.
    # This resizing and cropping process helps in maintaining the aspect ratio. 
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize tensors
    ])

    # Labels are automatically created for each folder
    train_dataset = datasets.ImageFolder(
        r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\raw\train', 
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\raw\valid', 
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\raw\test', 
        transform=transform
    )

    # Saves processed data to data/processed
    torch.save(train_dataset, r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\processed\train_dataset.pt')
    torch.save(val_dataset, r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\processed\val_dataset.pt')
    torch.save(test_dataset, r'C:\School\Minor\MLOPS\mlops-project-26\mlops-project-26\data\processed\test_dataset.pt')