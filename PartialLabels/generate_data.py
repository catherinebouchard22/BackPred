import numpy as np
import matplotlib.pyplot as plt

# xx and yy are 200x200 tables containing the x and y coordinates as values
# mgrid is a mesh creation helper
xx, yy = np.mgrid[:50, :50]
image = np.zeros((50,50))
label = np.zeros((50,50))

N = np.random.randint(1,30)

# Probability ofr a cell to be labeled
probability_of_label = 0.2

for i in range(N):
	x_center = np.random.randint(0,50)
	y_center = np.random.randint(0,50)
	distance = np.random.randint(10,20)
	value = np.clip(np.random.random(),0.2,1)

	# circles contains the squared distance to the (x_center, y_center) point
	circle = ((xx - x_center) ** 2 + (yy - y_center) ** 2) < distance
	image = image + circle*value

	if np.random.random() < probability_of_label:
		label[x_center-1:x_center+2, y_center] = 1
		label[x_center, y_center-1:y_center+2] = 1

image = image + np.random.poisson(size=(50,50))*0.05
plt.subplot(121)
plt.imshow(image,cmap='inferno')
plt.subplot(122)
plt.imshow(label,cmap='inferno')
plt.show()