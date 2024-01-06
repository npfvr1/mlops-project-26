import os
import torch
import torchvision
from torchvision import datasets, transforms


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    train_dataset = datasets.ImageFolder(r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\raw\train', transform=transform)
    val_dataset = datasets.ImageFolder(r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\raw\valid', transform=transform)
    test_dataset = datasets.ImageFolder(r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\raw\test', transform=transform)


    torch.save(train_dataset, r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\processed\train_dataset.pt')
    torch.save(val_dataset, r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\processed\val_dataset.pt')
    torch.save(test_dataset, r'C:\Users\Nima Jalili\Desktop\mlops-project-26-main\mlops-project-26\data\processed\test_dataset.pt')

    pass


test