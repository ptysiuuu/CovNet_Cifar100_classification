import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transformers import AutoConfig, AutoModelForImageClassification
import matplotlib.pyplot as plt
from train_epoch import train_epoch
from eval import eval
from cifar_dataloader import trainloader, c_testloader, val_loader
from early_stopper import EarlyStopping
from torch.optim.lr_scheduler import StepLR
from conv import Conv


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Conv()
load = True
save = False
load_path = 'models/best_model.pth'
save_path = 'models/best_model.pth'
max_epochs = 1
lr = 1e-3
criterion = nn.CrossEntropyLoss()


def main():
    if load:
        model = Conv()
        model.load_state_dict(torch.load(load_path, weights_only=True))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    epoch_losses = []
    train_losses = []
    val_losses = []
    early_stopper = EarlyStopping(patience=10, delta=0.01)
    epochs_completed = 0
    for epoch in range(max_epochs):
        epoch_loss, epoch_accuracy = train_epoch(model, criterion, optimizer, trainloader, device)
        epoch_losses.append(epoch_loss)
        train_losses.append(epoch_loss)
        val_accuracy, val_loss = eval(model, criterion, val_loader, device)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{max_epochs}] - Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}%")
        exp_lr_scheduler.step()
        epochs_completed += 1
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping")
            break

    early_stopper.load_best_model(model)
    eval_ret = eval(model, criterion, c_testloader, device)
    print(f"Test Accuracy: {eval_ret[0]*100:.2f}%, Test Loss: {eval_ret[1]:.4f}")
    if save:
        torch.save(model.state_dict(), save_path)

    plt.plot(range(1, epochs_completed + 1), train_losses, label="Train")
    plt.plot(range(1, epochs_completed + 1), val_losses, label="Val")
    plt.xlabel('Epoch', fontweight="bold")
    plt.ylabel('Loss', fontweight="bold")
    plt.title('Loss for epoch', fontweight="bold")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
