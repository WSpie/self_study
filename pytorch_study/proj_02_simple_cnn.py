import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


def save_checkpoint(state, checkpoint_path):
    print('==> Saving checkpoint.')
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path):
    print('==> Loading checkpoint.')
    model.load.state_dict(checkpoint_path['state_dict'])
    optimizer.load_state_dict(checkpoint_path['optimizer'])


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = False

checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = '_'.join(os.path.abspath(__file__).split('/')[-1].split('_')[:2])
checkpoint_path = f'{checkpoint_dir}/{checkpoint_prefix}_checkpoint.pth.tar'

# load data
train_dataset = datasets.MNIST(root='mnist_dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='mnist_dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# init network
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load(checkpoint_path))

# train the network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):

        # check if we can use cuda
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    
    checkpoint = {'epoch': epoch,
                 'state_dict': model.state_dict(), 
                 'optimizer': optimizer.state_dict(),
                 'loss': sum(losses) / len(losses)}
    save_checkpoint(checkpoint, checkpoint_path=checkpoint_path)

    print(f"Loss at epoch {epoch} was {sum(losses) / len(losses)}.")


def check_accuracy(loader, model):

    if loader.dataset.train:
        print('Checking training dataset')
    else:
        print('Checking test dataset')

    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    acc = round(float(num_correct) / float(num_samples) * 100, 2)
    print(f'Got {num_correct}/{num_samples} with accuracy {acc}% ')

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


