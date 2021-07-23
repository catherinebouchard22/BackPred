import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

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
model.load_state_dict(torch.load('mnist_14_model.pth', map_location='cpu'))

# Load test image
test_im = Image.open('mix31.png')
#test_im = torch.from_numpy(test_im).unsqueeze(0).unsqueeze(0)

# Over the whole validation dataset

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Predict class for test_im
test_im = transform(test_im).unsqueeze(0)
pred = model(test_im)
print(pred)

valset = datasets.MNIST('PATH_TO_STORE_TESTSET/MNIST', download=False, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=True)

def test():
  model.eval()
  val_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in valloader:
      output = model(data).squeeze()
      val_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

  val_loss /= len(valloader.dataset)

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    val_loss, correct, len(valloader.dataset),
    100. * correct / len(valloader.dataset)))

#test()