import numpy as np

a = np.array([[1, 0], [0, 1]])
b = np.array([[4, 1], [2, 2]])

print a*b

test_image = np.array([1,2,3]).T

train = np.array([[2,3,4], [4,5,6],[8,9,0],[8,3,2]])

print test_image
print train
print test_image.shape
print train.shape

distances = train - test_image
print distances
print np.sum(distances, axis=1)


