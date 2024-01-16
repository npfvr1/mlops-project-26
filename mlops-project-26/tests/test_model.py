import torch
from src.models.model import ResNet

# Inputs for the model
def model_inputs():
    batch_size = 32
    channels = 3
    height = 224
    width = 224
    num_classes = 100
    return torch.randn(batch_size, channels, height, width), torch.Size([batch_size, num_classes])

# Different input sizes for the model, height x width have to be with the same aspect ratio
def different_input_sizes():
    return [(224, 224), (256, 256), (299, 299)]

# Test model initialization and output shape
def test_model_init():
    model = ResNet()
    assert model.network.fc.out_features == 100, "The model does not have the expected number of output features."

# Test model output shape
def test_model():
    model = ResNet()
    xb = model_inputs()[0]
    out = model(xb)
    assert out.shape == model_inputs()[1], "The model does not produce the expected output shape."

# Test model training and evaluation modes
def test_model_modes():
    model = ResNet()
    model.train()
    assert model.training == True, "Model is not in training mode."
    model.eval()
    assert model.training == False, "Model is not in evaluation mode."

# Test model output for different input sizes
def test_model_different_input_sizes():
    model = ResNet()
    for height, width in different_input_sizes():
        xb = torch.randn(32, 3, height, width)
        out = model(xb)
        assert out.shape == torch.Size([32, 100]), f"Model output shape is incorrect for input size {height}x{width}."

# Test gradients flow through the model
def test_gradient_flow():
    model = ResNet()
    xb, _ = model_inputs()
    out = model(xb)
    out.mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradients for {name}"
        
# Test model consistency in forward pass for the same input        
def test_consistency_in_forward_pass():
    model = ResNet()
    model.eval()
    xb, _ = model_inputs()
    out1 = model(xb)
    out2 = model(xb)
    assert torch.equal(out1, out2), "Model outputs are not consistent for the same input."
        
