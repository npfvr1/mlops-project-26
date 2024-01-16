import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor to the chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model, dataset):
    device = get_default_device()
    model = to_device(model, device)
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]

def main():
    # Look into the directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data', 'processed')
    classes = os.listdir(data_dir + "/train_dataset.pt")

    # Data transforms (normalization & data augmentation)
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # PyTorch dataset
    dataset = ImageFolder(data_dir+'/train_dataset.pt', transform=transformations)

    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 525)

    # Load model weights
    model_path = './train_model'
    model.load_state_dict(torch.load(model_path))

    # Set model to evaluation mode
    model.eval()

    # Example usage
    test_dataset = ImageFolder(data_dir+'/test_dataset.pt', transform=transformations)
    img, label = test_dataset[100]
    plt.imshow(img.permute(1, 2, 0))
    predicted_label = predict_image(img, model, dataset)
    print('Label:', dataset.classes[label], ', Predicted:', predicted_label)

if __name__ == '__main__':
    main()