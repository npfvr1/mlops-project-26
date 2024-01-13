import torch
from src.data.make_dataset import *
from torchvision import datasets
import pytest

def load_dataset(file_path):
    # Load the dataset
    return torch.load(file_path)

@pytest.mark.parametrize("file_path", [
    "./data/processed/test_dataset.pt",
    "./data/processed/train_dataset.pt",
    "./data/processed/val_dataset.pt"
])
def test_dataset_shape_and_type(file_path):
    dataset = load_dataset(file_path)
    assert isinstance(dataset, datasets.ImageFolder), "Dataset type mismatch"

    for img, _ in dataset:
        # Check shape
        assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
        # Check data type
        assert img.dtype == torch.float32, f"Unexpected data type: {img.dtype}"


    
# Run the test
if __name__ == '__main__':
    test_dataset_shape_and_type()