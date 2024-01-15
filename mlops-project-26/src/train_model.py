import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train the model using the given data loader and optimizer.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        optimizer: The optimizer used for training.
        device: The device (CPU or GPU) on which the training will be performed.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch in train_loader:
        # Move data to the device
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss

# Example usage:

# Assuming you have defined the ResNet model, train_loader, and optimizer
# model = ResNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Move the model to the device
device = get_default_device()
model = model.to(device)

# Number of training epochs and learning rate
num_epochs = 5
lr = 0.001

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_loss = train(model, train_loader, optimizer, device)
    
    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
