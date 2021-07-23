import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])

trainset = torchvision.datasets.CIFAR10('PATH_TO_STORE_TRAINSET/CIFAR10', download=True, train=True, transform=transform)
valset = torchvision.datasets.CIFAR10('PATH_TO_STORE_TESTSET/CIFAR10', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)
        self.fc1 = nn.Conv2d(20, 10, kernel_size=4)
        self.fc2 = nn.Conv2d(10, 10, kernel_size=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15

gpu = torch.cuda.is_available()
if gpu:
	model.cuda()
	
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:

		if gpu:
			images = images.cuda()
			labels = labels.cuda()

		# Training pass
		optimizer.zero_grad()

		output = model(images).squeeze()
		loss = criterion(output, labels)

		#This is where the model learns by backpropagating
		loss.backward()

		#And optimizes its weights here
		optimizer.step()

		running_loss += loss.item()

		torch.save(model.state_dict(), './cifar10_{}_model.pth'.format(e)) 
	else:
		print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))


print("\nTraining Time (in minutes) =",(time()-time0)/60)