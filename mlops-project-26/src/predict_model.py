import argparse

import torch
from torchvision.datasets import ImageFolder

from models.model import ResNet
from data.make_dataset import get_transform


parser = argparse.ArgumentParser()
parser.add_argument('model',
                    action="store",
                    help="Path to pretrained model file (.pth)")
# The images must be inside a subfolder inside a folder because we use ImageFolder
parser.add_argument('img',
                    action="store",
                    help="Path to image folder (image format .jpg, RGB)")
args = parser.parse_args()


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


def predict(images, model, dataset):
    predicted_labels = []
    device = get_default_device()
    model = to_device(model, device)
    # Turn off gradients for prediction
    with torch.no_grad():
        # Set model to evaluation mode (faster inference)
        model.eval()
        for img in images:
            xb = to_device(img.unsqueeze(0), device)
            yb = model(xb)
            _, preds = torch.max(yb, dim=1)
            predicted_labels.append(dataset.classes[preds[0].item()])
    return predicted_labels


def main():
    # PyTorch dataset
    dataset = ImageFolder(r'data\raw\train')

    # Load pre-trained model
    model = ResNet()
    model.load_state_dict(torch.load(args.model))

    # Load the data to predict and process it in the same way as the training data
    images_to_predict = [i for i, _ in ImageFolder(args.img, transform=get_transform())]
    labels = predict(images_to_predict, model, dataset)
    print(labels)


if __name__ == '__main__':
    main()