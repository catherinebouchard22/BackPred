import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

# Define network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Conv2d(20, 10, kernel_size=4)
        self.fc2 = nn.Conv2d(10, 10, kernel_size=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

# Load trained weights
model.load_state_dict(torch.load('mnist_14_model.pth'))

# Load test image
test_im = np.load('results/1.npy')
test_im = torch.from_numpy(test_im).unsqueeze(0).unsqueeze(0)

# Predict class for test_im
pred = model(test_im)

print(pred)