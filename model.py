import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1x1_1 = nn.Conv2d(20, 10, 1)

        # Second block
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 20, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Final block
        self.conv5 = nn.Conv2d(20, 40, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(40)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(40, 10, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv1x1_1(x))

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Final block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
