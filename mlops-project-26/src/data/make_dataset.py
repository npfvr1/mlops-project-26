import os
import torch
from torchvision import datasets, transforms

def get_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform

def get_data_sets(transform):
    base_data_path = r'C:\Users\dongz\Desktop\02476\archive'

    train_dataset = datasets.ImageFolder(
        os.path.join(base_data_path, 'train'), 
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(base_data_path, 'valid'), 
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(base_data_path, 'test'), 
        transform=transform
    )   
    
    return [train_dataset, val_dataset, test_dataset]

def save_data_sets():    
    transform = get_transform()
    train_dataset, val_dataset, test_dataset = get_data_sets(transform)
    
    # Define the processed data path
    processed_data_path = r'C:\Users\dongz\Desktop\02476\mlops-project-26\mlops-project-26\src\data\processed'
    
    # Check if the directory exists, and create it if not
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # Saves processed data to data/processed
    torch.save(train_dataset, os.path.join(processed_data_path, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(processed_data_path, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(processed_data_path, 'test_dataset.pt'))

if __name__ == '__main__':
    save_data_sets()
