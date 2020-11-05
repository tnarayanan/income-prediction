import torch.nn as nn
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()  # input: 256x256
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2) # now: 128x128
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # now 64x64
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # now 32x32
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # now 16x16
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2) # now 8x8

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.relu(self.conv5(x))
        x = self.pool3(x)
        x = torch.relu(self.conv6(x))
        x = self.pool4(x)
        x = torch.relu(self.conv7(x))
        x = self.pool5(x)

        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x