import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# LeNet architecture
# input: 1*32*32 -> conv(size=5, stride=1, padding=0): 6*28*28 -> avg pool (2, s=2, p=0): 6*14*14 -> 
# conv(5, s=1, p=0): 16*10*10 -> avg pool(2, s=2, p=0): 16*5*5 -> FC 120 -> FC 84 -> Output: 10

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5))
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1) # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)