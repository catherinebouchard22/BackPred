import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import tifffile 

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

model = Net()

# Load trained weights
model.load_state_dict(torch.load('mnist_14_model.pth',map_location=torch.device('cpu')))

fig, axs = plt.subplots(2,5)
for label in range(10):
    print('Computing for class {}'.format(label))

    # Load test image
    test_im = np.load('results/{}.npy'.format(label))

    test_im = torch.from_numpy(test_im).unsqueeze(0).unsqueeze(0)
    test_im = (test_im - test_im.min()) / (test_im.max() - test_im.min())

    # Variability map
    var_map = np.zeros((28,28))
    pred_map = np.zeros((28,28))
    acc_map = np.zeros((28,28)) 

    # Iterate over each pixel
    for x in range(28):
        for y in range(28):

            pixel_var = 0
            pixel_pred = 0
            pixel_acc = 0
            for value in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

                # Change the value of the pixel
                test_im_modified = test_im.clone()
                test_im_modified[0,0,x,y] = value

                # Predict class for test_im_modified
                pred = model(test_im_modified).detach().numpy()[0]

                # Normalize the prediction
                pred = softmax(pred)

                # Add the difference between the class and all others to the pixel variability
                for el in pred:
                    pixel_var += (pred[label] - el) / 10

                pixel_pred += np.argmax(pred) / 11
                pixel_acc += int(np.argmax(pred)==label) / 11

            var_map[x,y] = pixel_var
            pred_map[x,y] = pixel_pred
            acc_map[x,y] = pixel_acc

    np.save('{}_var.npy'.format(label), var_map)
    tifffile.imsave('{}_var.tif'.format(label), var_map)
    tifffile.imsave('{}_pred.tif'.format(label), pred_map)
    tifffile.imsave('{}_acc.tif'.format(label), acc_map)
    axs.ravel()[label].imshow(var_map, cmap='Reds')
    axs.ravel()[label].set_title(label)

plt.show()
