import numpy as np
import numpy.ma as ma

X = np.array([[1,2,3], [2,3,4]])

train = np.array([[2,3,4], [4,5,6],[8,9,0],[8,3,2]])


W = np.zeros([10,3])
print "W shape = {}".format(W.shape)

# 2 images, 3 classes
scores = np.array([[2.3, 4.5, -1.0],
                  [1.5, 0.5, 10.0]])

y = np.array([0, 2])
classes = np.array([0, 1, 2], dtype=int)

print "scores shape = {}, y shape = {}, classes shape = {}".format(scores.shape, y.shape, classes.shape)

mask_temp = np.zeros(scores.shape, dtype=int)
mask_temp[:, None] = classes
print mask_temp
y_mask = np.repeat(y, scores.shape[1]).reshape(scores.shape)
print y_mask
mask = ma.array(scores, mask=mask_temp==y_mask)
print mask

