import torch
from src.data.make_dataset import *
from torchvision import datasets
import pytest

def get_expected_lengths():
    
    expected_len_test__dataset = 2625
    expected_len_train__dataset = 84635
    expected_len_val__dataset = 2625
    
    return [expected_len_test__dataset, expected_len_train__dataset, expected_len_val__dataset]

def test_loading_data():
    
    transform = get_transform()
    datasets = get_data_sets(transform)
    train_dataset, val_dataset, test_dataset = datasets
    
    expected_lengths = get_expected_lengths()
    assert len(test_dataset) ==  expected_lengths[0], "The train dataset is loaded incorrectly"
    assert len(train_dataset) ==  expected_lengths[1], "The validation dataset is loaded incorrectly"
    assert len(val_dataset) ==  expected_lengths[2], "The test dataset is loaded incorrectly"


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
    test_loading_data()
    test_dataset_shape_and_type()