import torch.nn as nn
# import numpy as np
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)  # 5 x 252 x 252
        self.pool1 = nn.MaxPool2d(2, 2)  # 5 x 126 x 126
        self.conv2 = nn.Conv2d(5, 10, 5)  # 10 x 122 x 122
        self.pool2 = nn.MaxPool2d(2, 2)  # 10 x 61 x 61
        self.conv3 = nn.Conv2d(10, 20, 4)  # 20 x 58 x 58
        self.pool3 = nn.MaxPool2d(2, 2)  # 20 x 29 x 29
        self.fc = nn.Linear(20 * 29 * 29, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 20 * 29 * 29)
        x = self.fc(x)
        return x
