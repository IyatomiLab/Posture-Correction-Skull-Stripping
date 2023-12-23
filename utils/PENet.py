import torch
from torch import nn


class PENet(nn.Module):
    def __init__(self, out_channels: int):
        super(PENet, self).__init__()
        self.conv0 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.fc0 = nn.Linear(4*4*32, 128)
        self.fc1 = nn.Linear(128, out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.relu(self.conv0(x)))
        h = self.pool(self.relu(self.conv1(h)))
        h = self.pool(self.relu(self.conv2(h)))
        h = h.view(h.size(0), -1)
        h = self.relu(self.fc0(h))
        h = self.fc1(h)
        return h
