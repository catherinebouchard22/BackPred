# Source: https://github.com/utkuozbulak/pytorch-cnn-visualizations
# Source: https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

import os
import copy
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import models
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import PIL.ImageOps

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.float32(recreated_im).transpose(1, 2, 0)
    return recreated_im

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, selected_filter):
        self.model = Net()
        self.model.load_state_dict(torch.load('mnist_14_model.pth', map_location='cpu'))
        #self.model.cuda().eval()
        self.model.eval()
        self.selected_filter = selected_filter
        self.conv_output = 0

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model.fc2.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        for j in range(20):
            self.hook_layer()
            # Generate a random image
            sz = 28
            random_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 1)))
            # Process image and return variable
            processed_image = preprocess_image(random_image, False)
            # Define optimizer for the image
            optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
            for i in range(1, 30):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                x = processed_image#.cuda(0)

                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = self.model(x)
                # Only need to forward until the selected layer is reached

                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                loss = -torch.mean(self.conv_output)
                # Backward
                loss.backward()
                # Update image
                optimizer.step()

            # Recreate image
            #self.created_image = recreate_image(processed_image)
            image = np.float32(processed_image.cpu().detach().squeeze())
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
            self.created_image = PIL.ImageOps.invert(Image.fromarray(image))
            self.created_image = self.created_image.resize((280,280), Image.NEAREST)
            #self.created_image.save('iterative/{}_{}.png'.format(self.selected_filter, i))
            self.created_image.save('{}s/{}_{}.png'.format(self.selected_filter, self.selected_filter, j))

# Modèle
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


for filter_pos in range(10):
    os.mkdir('{}s'.format(filter_pos))
    layer_vis = CNNLayerVisualization(filter_pos)
    layer_vis.visualise_layer_with_hooks()