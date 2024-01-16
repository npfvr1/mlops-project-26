import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to the chosen device"""
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
    data_dir = os.path.join(current_dir, 'data', 'processed', 'test_dataset.pt')
    classes = os.listdir(data_dir + "/train")

    # Data transforms (normalization & data augmentation)
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # PyTorch dataset
    dataset = ImageFolder(data_dir+'/train', transform=transformations)

    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 525)

    # Load model weights
    model_path = ''
    model.load_state_dict(torch.load(model_path))

    # Set model to evaluation mode
    model.eval()

    # Example usage
    img_path = ''
    img = Image.open(img_path)
    img = transformations(img)
    label = predict_image(img, model, dataset)
    print('Predicted label:', label)

if __name__ == '__main__':
    main()