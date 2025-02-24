import tqdm
import torch


def train_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    model.to(device)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm.tqdm(loader, desc='Training', ncols=100):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, targets)
        loss.to(device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy
