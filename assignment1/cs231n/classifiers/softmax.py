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
  num_train = y.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
      logits = X[i].dot(W)
      logits_norm = logits - np.max(logits)
      logits_norm_exp = np.exp(logits_norm)
      logits_norm_exp_sum = np.sum(logits_norm_exp)
      logits_norm_exp_sum_log = np.log(logits_norm_exp_sum)

      loss += -1.0*logits_norm[y[i]] + logits_norm_exp_sum_log

      # gradient computing (dW = d_loss/d_logits * X):
      d_logits_norm_y_i = -1.0                   # d_loss/d_logits_norm[y[i]]
      d_logits_norm_exp_sum_log = 1              # d_loss/d_logits_norm_exp_sum_log
      d_logits_norm_exp_sum = d_logits_norm_exp_sum_log * (1/logits_norm_exp_sum)  # d_loss/d_logits_norm_exp_sum_log * d_logits_norm_exp_sum_log/d_logits_norm_exp_sum
      d_logits_norm_exp = d_logits_norm_exp_sum * np.ones_like(logits_norm_exp)
      d_logits_norm = d_logits_norm_exp * logits_norm_exp
      d_logits_norm[y[i]] += d_logits_norm_y_i
      d_logits = d_logits_norm                   # d_loss/d_logits

      dW += np.dot((X[i].T).reshape(-1,1), d_logits.reshape(1,-1))

  loss /= num_train
  loss += reg * np.sum(W**2)

  dW /= num_train
  dW += 2 * reg * W

  a = """
  for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      prob = np.exp(scores) / np.sum(np.exp(scores))
      loss += -np.log(prob[y[i]])

      for j in range(num_classes):
          if j == y[i]:
              dW[:, j] -=  (1 - prob[j]) * X[i]
          else:
              dW[:, j] -= -prob[j] * X[i]

  #print('dW: ', dW)
  #print('loss: ', loss)
  #raise SystemExit

  loss /= num_train
  dW /= num_train

  # Add regularisation:
  loss += reg * np.sum(W ** 2)
  dW += 2 * reg * W
  
  """
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
  num_train = y.shape[0]
  num_classes = W.shape[1]

  logits = X.dot(W)
  logits_norm = logits - np.max(logits, axis=1).reshape(-1,1)
  logits_norm_exp = np.exp(logits_norm)
  logits_norm_exp_sum = np.sum(logits_norm_exp, axis=1)
  logits_norm_exp_sum_log = np.log(logits_norm_exp_sum)

  loss = -1.0 * logits_norm[np.arange(num_train), y] + logits_norm_exp_sum_log

  # gradient computing (dW = d_loss/d_logits * X)
  d_logits_norm_y = -1.0
  d_logits_norm_exp_sum_log = 1
  d_logits_norm_exp_sum = d_logits_norm_exp_sum_log * (1/logits_norm_exp_sum)
  d_logits_norm_exp = d_logits_norm_exp_sum.reshape(-1,1) * np.ones_like(logits_norm_exp)
  d_logits_norm = d_logits_norm_exp * logits_norm_exp
  d_logits_norm[np.arange(num_train), y] += d_logits_norm_y
  d_logits = d_logits_norm

  dW = np.dot(X.T, d_logits)
  dW /= num_train
  dW += 2*reg*W

  loss = np.sum(loss) / num_train


  a = """
  scores = X.dot(W)
  scores -= (scores.max(axis=1)).reshape(num_train, 1)
  probs = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)

  target_probs = probs[np.arange(len(probs)), y]
  loss = np.sum(-np.log(target_probs)) / num_train
  #loss += reg * np.sum(W**2)

  #1) i == j:
  mask = np.zeros_like(scores)
  mask[np.arange(mask.shape[0]), y] = 1 - target_probs
  dW1 = np.dot(X.T, mask)

  #2) i != j:
  mask = probs
  mask[np.arange(mask.shape[0]), y] = 0
  dW2 = np.dot(X.T, mask)

  dW = (-dW1 + dW2) / num_train
  
  # Add regularisation:
  loss += reg * np.sum(W ** 2)
  dW += 2 * reg * W
  """



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

