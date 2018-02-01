import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # calculate the loss, one sample by another
  for i in range(X.shape[0]):
      scores = np.dot(X[i], W)
      # shift the values to avoid the numeric problem
      # the highest value would be zero
      scores -= np.max(scores)

      # calculate the negative log probability
      probs = np.exp(scores)
      sum_probs = np.sum(probs)
      loss_vec = - np.log(probs / sum_probs)

      # retrieve the negative log probability of the predicted class
      correct_label_index = y[i]
      loss += loss_vec[correct_label_index]

      # References:
      # https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function
      # https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
      # calculate the gradients of weights
      # dW = mean( (pi - yi) * Xi )
      #   pi:  probability vector for the i example
      #   yi:  the indicator function, indicates the actual target class
      #   Xi:  the input of the i example
      for c in range(W.shape[1]):
          dW[:, c] +=  ( (probs / sum_probs)[c] - (c == y[i]) ) * X[i,:]

  # average the loss over all examples
  loss = loss / X.shape[0]

  # add the regularization
  loss += 0.5 * reg * np.sum(W * W)

  # average the gradients over the examples
  dW = dW / X.shape[0]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # calculate the loss, all at once
  
  scores = np.dot(X, W)    # output shape: (N, C)
  # shift the values to avoid the numeric problem
  # the highest value would be zero
  scores -= np.max(scores, axis=1).reshape(len(scores), 1)

  # calculate the negative log probability
  probs = np.exp(scores)
  sum_probs = np.sum(probs, axis=1).reshape(len(probs), 1)  # reshape to (N, C)
  loss_matrix = - np.log(probs / sum_probs)

  # retrieve the negative log probability of the predicted class
  correct_label_index = y  # just to facilitate the reading of code
  loss += np.sum(loss_matrix[np.arange(len(loss_matrix)), correct_label_index])

  # calculate the gradients of weights
  # dW = mean( (pi - yi) * Xi )
  #   pi:  probability vector for the i example
  #   yi:  the indicator function, indicates the actual target class
  #   Xi:  the input of the i example
  
  # mark the selected class for each example
  indicators = np.zeros_like(probs)
  indicators[np.arange(len(indicators)), y] = 1
  
  dW += np.dot( X.T, (probs/sum_probs)-indicators )
  
  # average the loss over all examples
  loss = loss / X.shape[0]

  # add the regularization
  loss += 0.5 * reg * np.sum(W * W)

  # average the gradients over the examples
  dW = dW / X.shape[0]
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

