import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from cifar_dataloader import c_testloader
from backbone_dataloader import testloader as b_testloader
from backbone_dataloader import trainloader as b_trainloader
from backbone_dataloader import val_loader as b_val_loader
from train_epoch import train_epoch
from eval import eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "conv_10_3_v14.pth"
resnet_path = "resnet18v1.pth"
criterion = nn.CrossEntropyLoss()

def main():
    my_conv_model = torch.load(path)
    my_conv_model.to(device)
    my_conv_model.eval()

    resnet18 = models.resnet18(weights=None)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )
    resnet18.load_state_dict(torch.load(resnet_path))
    resnet18.to(device)

    models_dict = {
        "Nasz model": [my_conv_model, c_testloader],
        "ResNet-18": [resnet18, b_testloader],
    }

    accuracies = {}
    for name, net in models_dict.items():
        acc, loss = eval(net[0], criterion, net[1], device)
        acc = acc * 100
        accuracies[name] = acc
        print(f"{name}: {acc:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values(),
            color=['blue', 'red'])
    plt.xticks(rotation=15, ha='right')
    plt.ylabel("Dokładność [%]")
    plt.title("Porównanie dokładności modeli na CIFAR-100 (test set)")
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

