import torch
import torch.nn as nn

from torchvision.models import resnet50, efficientnet_b4, densenet121, vit_b_16, inception_v3

class ConvNet(nn.Module):
    def __init__(self, 
                 num_classes,
                 num_features,
                 num_in_features,
                 num_out_features):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(num_in_features, num_features, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
        self.drop1 = nn.Dropout(0.3)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        # Fully connected layers
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(num_out_features, 512)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, num_classes)
 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.relu(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x

# Architecture for MNIST and FashionMNIST
def mnist_net():
    return ConvNet(num_classes=10, num_features=32, num_in_features=1, num_out_features=6272)

# Architecture for CIFAR10
def cifar10_net():
    return ConvNet(num_classes=10, num_features=32, num_in_features=3, num_out_features=8192)

# Architecture for CelebA
def celebA_net():
    return ConvNet(num_classes=40, num_features=32, num_in_features=3, num_out_features=8192)
