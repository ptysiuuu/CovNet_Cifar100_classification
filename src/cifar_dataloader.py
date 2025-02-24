import torch
import torchvision
import torchvision.transforms as transforms

data_transforms = {
    'train': transforms.Compose([
     transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15),
     transforms.RandomCrop(32, padding=4)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 256

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=data_transforms['train']
)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=data_transforms['val']
)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

c_testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=2
)
