import numpy
import tifffile

for i in range(10):
	im_np = numpy.load('{}.npy'.format(i))
	im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min()) * 255
	tifffile.imsave('{}'.format(i), im_np)