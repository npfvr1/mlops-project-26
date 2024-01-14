import torch
from torchvision import datasets, transforms

def get_transform():
    
    # Adjusted transform by resizing to 256 (was before 224), and centercrop to 224, 
    # which is a standard approach for preparing images for models like ResNet.
    # This resizing and cropping process helps in maintaining the aspect ratio.  
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize tensors
    ])
    return transform


def get_data_sets(transform):
    
    # Labels are automatically created for each folder
    train_dataset = datasets.ImageFolder(
        r'.\data\raw\train', 
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        r'.\data\raw\valid', 
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        r'.\data\raw\test', 
        transform=transform
    )   
    
    return [train_dataset, val_dataset, test_dataset]

def save_data_sets():    
    transform = get_transform()
    train_dataset, val_dataset, test_dataset = get_data_sets(transform)
    
    # print(f"Data set = {len(test_dataset)}")  

    # Saves processed data to data/processed
    torch.save(train_dataset, r'.\data\processed\train_dataset.pt')
    torch.save(val_dataset, r'.\data\processed\val_dataset.pt')
    torch.save(test_dataset, r'.\data\processed\test_dataset.pt')


if __name__ == '__main__':
    save_data_sets()            