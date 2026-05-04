import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #first convolution layer
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        #second convolution layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        #pooling layer (size reduction, keep important features)
        self.pool = nn.MaxPool2d(2, 2)
        #connected layers used for classification
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #apply convolution + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #flatten image into a vector
        x = x.view(-1, 64 * 8 * 8)
        #pass through connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x