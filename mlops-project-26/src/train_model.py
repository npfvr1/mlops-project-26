import torch
from torch.utils.data.dataloader import DataLoader, Dataset
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from models.model import ResNet


'''Run this script to train a new Network on the data under ./data/processed
The trained network will be saved as a .pt file under ./models/
The training curves will be saved under ./reports/figures
The hyperparameters used will be logged under ./outuputs/[new dated folder]
'''


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.clf()
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(r'./reports/figures/accuracy.png')


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.clf()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(r'./reports/figures/loss.png')


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        i = 0
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(len(train_loader), i, len(batch[0]), len(batch[1]))
            i+=1
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        plot_accuracies(history)
        plot_losses(history)

    torch.save(model.state_dict(), r'models/checkpoint.pth')
    

@hydra.main(version_base=None, config_path="../", config_name="config")
def train(cfg: DictConfig) -> None:
    train_ds = torch.load(r'.\data\processed\train_dataset.pt')
    test_ds = torch.load(r'.\data\processed\test_dataset.pt')
    train_dl = DataLoader(train_ds,
                          cfg.hyperparameters.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
    test_dl = DataLoader(test_ds,
                         cfg.hyperparameters.batch_size,
                         num_workers=4,
                         pin_memory=True)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    model = to_device(ResNet(), device)
    fit(cfg.hyperparameters.epochs,
        cfg.hyperparameters.lr,
        model,
        train_dl,
        test_dl)

if __name__ == "__main__":
    train()