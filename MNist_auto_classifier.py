import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt

# https://nextjournal.com/gkoehler/pytorch-mnist

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

valset = datasets.MNIST('PATH_TO_STORE_TESTSET/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)

answers = []
n = 7
b = 28//n
correct, not_correct = 0, 0
for i in range(10):
	answer = tifffile.imread('results/{}.tif'.format(i))
	answer = answer.reshape(-1, n, b, n).sum((-1, -3)) / n
	answer = (answer - answer.min()) / (answer.max() - answer.min())
	answers.append(answer)

for images, labels in valloader:
	images = images.reshape(-1, n, b, n).sum((-1, -3)) / n
	images = images.detach().numpy()
	iamges = (images - images.min()) / (images.max() - images.min())

	diff_all = []

	for i, answer in enumerate(answers):
		diff = np.mean(np.abs(images - answer))
		diff_all.append(diff)
	if labels.item() == np.argmin(diff_all):
		correct += 1
	else:
		not_correct += 1

print('{}/{} ({}%) were classified correctly'.format(correct, correct+not_correct, (correct/(correct+not_correct))*100))