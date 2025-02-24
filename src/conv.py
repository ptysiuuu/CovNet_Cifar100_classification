import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)

        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024 * 4 * 4, 4000)
        self.bn_fc1 = nn.BatchNorm1d(4000)
        self.fc2 = nn.Linear(4000, 800)
        self.bn_fc2 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800, 100)

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(self.pool(F.relu(self.bn6(self.conv6(x)))))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout_conv(self.pool(F.relu(self.bn9(self.conv9(x)))))
        x = F.relu(self.bn10(self.conv10(x)))

        x = torch.flatten(x, 1)
        x = self.dropout_fc(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x
