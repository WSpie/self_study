import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# Then flatten and Fc 4096 -> 4096 -> 1000

class VGG_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_Net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(4096, num_classes)

        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                    kernel_size=(3, 3), padding=(1, 1)), 
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        
        return nn.Sequential(*layers)

x = torch.randn(1, 3, 224, 224)
model = VGG_Net()
print(model(x).shape)