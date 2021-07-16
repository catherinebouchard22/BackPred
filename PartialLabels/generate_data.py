import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Probability ofr a cell to be labeled
probability_of_label = 0.5
sz = 64

for im in range(10000):
	# xx and yy are 200x200 tables containing the x and y coordinates as values
	# mgrid is a mesh creation helper
	xx, yy = np.mgrid[:sz, :sz]
	image = np.zeros((sz,sz))
	label = np.zeros((sz,sz))

	N = np.random.randint(0,5)

	for i in range(N):
		x_center = np.random.randint(0,sz)
		y_center = np.random.randint(0,sz)
		distance = np.random.randint(10,30)
		value = np.clip(np.random.random(),0.5,0.8) * 10 

		# circles contains the squared distance to the (x_center, y_center) point
		circle = ((xx - x_center) ** 2 + (yy - y_center) ** 2)
		circle = circle * (circle < distance)
		image = image + circle*value

		if np.random.random() < probability_of_label:
			# Points
			#label[x_center, y_center] = 255
			# Crosses
			label[x_center-1:x_center+2, y_center] = 255
			label[x_center, y_center-1:y_center+2] = 255
			# Circles
			#label = label + (((xx - x_center) ** 2 + (yy - y_center) ** 2) < 10)*255

	image = image + np.random.poisson(size=(sz,sz))*0.05*255

	image = np.clip(image, 0, 255)
	image = np.expand_dims(image,0)
	label = np.expand_dims(label,0)
	combined = np.concatenate([image, label]).astype('uint8')

	tifffile.imsave('train50/{}.tif'.format(im), combined)