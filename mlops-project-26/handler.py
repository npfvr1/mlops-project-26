import torch
from torchvision import transforms
from PIL import Image
import io
from src.data.make_dataset import get_transform

# Load the model
model = torch.jit.load("checkpoint.pth")
model.eval()

def transform_image(image_bytes):
    transform = get_transform()
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def handle(data, context):
    image = transform_image(data)
    result = model(image)
    return result.tolist()