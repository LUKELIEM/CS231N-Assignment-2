import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, verbose=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype  
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    
    # Set up weights for the filters of the CONV layer
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    
    # Calculate dimension of the 2x2 max pool layer
    # The filter size, stride and padding of the CONV layer is assumed to preserve the dimension
    # of the input volume
    maxpool_dim = num_filters * H * W / 4
 
    # Set up weights for the filters of the 2 FC layers
    self.params['W2'] = weight_scale * np.random.randn(maxpool_dim, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    
    if verbose:
        print self.params['W1'].shape
        print self.params['b1'].shape
        print self.params['W2'].shape
        print self.params['b2'].shape
        print self.params['W3'].shape
        print self.params['b3'].shape
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)  # Cast the datatype of params to be np.float32
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    # These parameters allows the activation map to have the same HxW dimension
    # as the input volume
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    # These parameters downsamples the activation map by 2 --> H/2 x W/2
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    reg_loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # Forward Pass - Stage 1 - conv - relu - 2x2 max pool                           [1]
    out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    reg_loss += np.sum(W1*W1)  # Accumulate the Regulatization Lossed
    
    # Forward Pass - Stage 2 - affine - relu                                        [2]
    out, affine_relu_cache = affine_relu_forward(out, W2, b2) 
    reg_loss += np.sum(W2*W2)  # Accumulate the Regulatization Lossed

    # Forward Pass - Stage 3 - affine                                               [3]
    scores, affine_cache = affine_forward(out, W3, b3)  # Forward pass - affine
    reg_loss += np.sum(W3*W3)  # Accumulate the Regulatization Lossed
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)                  # Forward and Backward pass - softmax
    loss += 0.5*self.reg*reg_loss                           # Regularization - forward pass


    

    # Backward Pass - Stage 3 - affine                                               [3]
    dout, dW3, db3 = affine_backward(dscores, affine_cache)
    
    # Backward Pass - Stage 2 - affine - relu                                        [2]
    dout, dW2, db2 = affine_relu_backward(dout, affine_relu_cache)

    # Backward Pass - Stage 1 - conv - relu - 2x2 max pool                           [1]
    __, dW1, db1 = conv_relu_pool_backward(dout, conv_relu_pool_cache)

    # Regularization - backward pass
    dW1 += self.reg * W1 
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    
    # Store into grads dictionary
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
