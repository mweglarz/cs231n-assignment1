import numpy as np

x = np.array([2,3,3,2,6])
print x
a = np.bincount(x)
print a
print np.argmax(a)

