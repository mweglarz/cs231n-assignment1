import numpy as np

X = np.array([[1,2,3], [2,3,4]])

train = np.array([[2,3,4], [4,5,6],[8,9,0],[8,3,2]])


print X
print train
print X.shape
print train.shape
dist = train[:, None] - X

print "************"

print (X**2).sum(axis=1, keepdims=True)


