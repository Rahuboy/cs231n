from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_examples = X.shape[0]

    
    for i in range(num_examples):
      scores = X[i].dot(W)
      class_probability = np.exp(scores[y[i]])
      cumulative_probability = 0
      for j in range(W.shape[1]):
        cumulative_probability += np.exp(scores[j])
      normalized_probability = class_probability/cumulative_probability
      loss += -1*np.log(normalized_probability)
      for j in range(W.shape[1]):
        dW[:, j] += X[i]*np.exp(scores[j])/cumulative_probability
        if(y[i] == j):
          dW[:, j] -= X[i]

    

    loss /= num_examples
    dW /= num_examples


    loss += reg*np.sum(W*W)
    dW += 2*reg*W
        
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_examples = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #loss
    scores = X @ W
    exp_scores = np.exp(scores)
    cumulative_exp_scores = np.sum(exp_scores, axis = 1)
    softmax = (exp_scores.T / cumulative_exp_scores).T
    class_probability = softmax[np.arange(num_examples), y]
    loss += np.sum(-1*np.log(class_probability))


    #gradient
    
    softmax[np.arange(num_examples), y] -= 1
    dW += (X.T @ softmax)


    loss /= num_examples
    dW /= num_examples


    loss += reg*np.sum(W*W)
    dW += 2*reg*W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
