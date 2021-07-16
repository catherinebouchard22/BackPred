import numpy
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

loss = numpy.load('cells_10000iterations_1prob_model.npy')

plt.plot(loss)
plt.plot(running_mean(loss,100), 'k')
plt.show()