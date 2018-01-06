import numpy as np
import numpy.ma as ma
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)

    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    classes = np.arange(10, dtype=int)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    print "W shape = {}".format(W.shape)
    print "X shape = {}".format(X.shape)
    print "y shape = {}".format(y.shape)
    print "classes shape = {}".format(classes.shape)
    # W.shape[1] numbers of images
    scores = X.dot(W)
    print "scores shape = {}".format(scores.shape)
    mask_1 = np.zeros(scores.shape, dtype=int)
    mask_1[:, None] = classes
    y_mask = np.repeat(y, scores.shape[1]).reshape(scores.shape)
    mask = mask_1 == y_mask

    correct_class_score = ma.array(scores, mask = ~mask)
    print "correct class score 0".format(correct_class_score)
    correct_class_score = np.sum(correct_class_score, axis=0)
    scores = ma.array(scores, mask = mask)
    # todo: create margin array and sums
    # todo: correct class score matrix
    print "correct class score shape = {}".format(correct_class_score.shape)
    print "y shape = {}".format(y.shape)
    print "scores = {}".format(scores[:1])
    # print "correct class score = {}".format(correct_class_score[:1])
    print "y = {}".format(y[:1])
    print "mask shape = {}".format(mask.shape)
    print "mask = {}".format(mask[:1])

    # margin = scores - correct_class_score + 1
    # print "margin shape = {}".format(margin.shape)
    # print "margin = {}".format(margin[:1])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

