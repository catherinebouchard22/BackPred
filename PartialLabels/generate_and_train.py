import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import tifffile 

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        return x

model = Net()

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
iterations = 10000

gpu = torch.cuda.is_available()
if gpu:
    model.cuda()
    
# Probability ofr a cell to be labeled
probability_of_label = 0.5
running_loss = []

for e in range(iterations+1):
    if e % 1000 == 0:
        print('{}/{}'.format(e, iterations))
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[:50, :50]
    images = np.zeros((50,50))
    labels = np.zeros((50,50))

    N = np.random.randint(1,30)

    for i in range(N):
        x_center = np.random.randint(0,50)
        y_center = np.random.randint(0,50)
        distance = np.random.randint(10,20)
        value = np.clip(np.random.random(),0.2,1)

        # circles contains the squared distance to the (x_center, y_center) point
        circle = ((xx - x_center) ** 2 + (yy - y_center) ** 2) < distance
        images = images + circle*value

        if np.random.random() < probability_of_label:
            # Circles
            #labels = labels + (((xx - x_center) ** 2 + (yy - y_center) ** 2) < 5)
            # Points
            labels[x_center, y_center] = 1 
            # Crosses
            #labels[x_center-1:x_center+2, y_center] = 1
            #labels[x_center, y_center-1:y_center+2] = 1

    images = images + np.random.poisson(size=(50,50))*0.02
    images = torch.tensor(images).unsqueeze(0).unsqueeze(0).type(torch.float)
    labels = torch.tensor(labels).unsqueeze(0).unsqueeze(0).type(torch.float)

    if gpu:
        images = images.cuda()
        labels = labels.cuda()

    # Training pass
    optimizer.zero_grad()

    output = model(images)
    loss = criterion(output, labels)

    #This is where the model learns by backpropagating
    loss.backward()

    #And optimizes its weights here
    optimizer.step()

    #Running loss
    running_loss.append(loss.item())

np.save('./cells_{}iterations_{}prob_model'.format(e,probability_of_label), running_loss) 
torch.save(model.state_dict(), './cells_{}iterations_{}prob_model.pth'.format(e,probability_of_label)) 

print("\nTraining Time (in minutes) =",(time()-time0)/60)