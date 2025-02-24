import torch


def eval(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss
