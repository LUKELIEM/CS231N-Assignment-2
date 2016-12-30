import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

 
class DeepConvNet(object):
  """
  A multi-layer convolutional network with an arbitrary number of CONV layers 
  with the following architecture:
  
  {conv - relu - max pool} x (L - 1) - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, num_filters=[32, 32], filter_sizes=[3, 3], input_dim=(3,32,32), hidden_dim=100, 
               num_classes=10, reg=0.0, weight_scale=1e-2, dtype=np.float32, 
               verbose=False):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - num_filters: A list of integers giving the number of filters in the CONV 
      layers.
    - filter_sizes: A list of integers giving the dimensions of the filters in 
      the CONV layers.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.reg = reg
    self.num_conv_layers = len(num_filters)
    self.dtype = dtype
    self.params = {}
    CONVout_dims = {}
    maxpool_dims = {}   

    ############################################################################
    # TODO: Initialize weights and biases for the multi-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the 1st convolutional layer using the keys  #
    # 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the  #
    # the 2nd convolutional layer, and keys 'W3' and 'b3' for the weights and  #
    # of the next convolutional of affine layer, and so on                     #
    ############################################################################
    num_conv_layers = self.num_conv_layers
    C, H, W = input_dim
    
    # Assign weight and biases for CONV layers
    for layer in range(num_conv_layers):
        if layer is 0:
            filter_depth = C  # CONV Layer 1 has same depth as input depth
            # In this configuration, there is 2x2 max pooling after each CONV layer, so there
            # is a 2-to-1 downsampling. In layer 0, it simply downsample the input dimensions.
            CONVout_dims[layer] = (num_filters[layer], H, W)
            maxpool_dims[layer] = (num_filters[layer], H/2, W/2)
        else:
            # Depth of other CONV Layers 1 has the same depth as number of filters in the preceding 
            # CONV layer
            filter_depth = num_filters[layer-1]
            # In this configuration, there is 2x2 max pooling after each CONV layer, so there
            # is a 2-to-1 downsampling. In subsequent layer, it simply downsample the dimensions
            # of the preceeding CONV layer output
            __, HH, WW = maxpool_dims[layer-1]
            CONVout_dims[layer] = (num_filters[layer], HH, WW)
            maxpool_dims[layer] = (num_filters[layer], HH/2, WW/2)
        
        # Set up weights for the filters of the CONV layer
        self.params[(layer,'W')] = weight_scale * np.random.randn(num_filters[layer],                                                     
                                                                  filter_depth, filter_sizes[layer], filter_sizes[layer])
        self.params[(layer,'b')] = np.zeros(num_filters[layer])
    
    # Assign weight and biases for FC layers (num_layer and num_layer+1)
    C, H, W = maxpool_dims[num_conv_layers-1]
    self.params[(num_conv_layers,'W')] = weight_scale * np.random.randn(C*H*W, hidden_dim)
    self.params[(num_conv_layers,'b')] = np.zeros(hidden_dim)

    self.params[(num_conv_layers+1,'W')] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params[(num_conv_layers+1,'b')] = np.zeros(num_classes)
  
    if verbose:
        for layer in range(num_conv_layers):
            print "W & b in CONV layer %d" % (layer+1)
            print self.params[(layer,'W')].shape
            print self.params[(layer,'b')].shape
            print "CONV output dimension: %d x %d x %d" % CONVout_dims[layer]
            print "Maxpool dimension: %d x %d x %d" % maxpool_dims[layer]  
        print "W & b in FC layers:"
        print self.params[(num_conv_layers,'W')].shape
        print self.params[(num_conv_layers,'b')].shape   
        print self.params[(num_conv_layers+1,'W')].shape
        print self.params[(num_conv_layers+1,'b')].shape  
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None, verbose=False):
    """
    Evaluate loss and gradient for the multi-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    W = {}
    b = {}
    dW = {}
    db = {}
    conv_params = {}
    conv_relu_pool_cache = {}
    num_conv_layers = self.num_conv_layers
    
    for i in range(num_conv_layers):
        # Retrieve weights and biases for CONV layers
        W[i], b[i] = self.params[(i,'W')], self.params[(i,'b')]
        # pass conv_param to the forward pass for the convolutional layer
        # These parameters allows the activation map to have the same HxW dimension
        # as the input volume
        filter_size = W[i].shape[2]
        conv_params[i] = {'stride': 1, 'pad': (filter_size - 1) / 2}
   
    # pass pool_param to the forward pass for the max-pooling layer
    # These parameters downsamples the activation map by 2 --> H/2 x W/2
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Retrieve weights and biases for FC layers
    W[num_conv_layers], b[num_conv_layers] = self.params[(num_conv_layers,'W')], self.params[(num_conv_layers,'b')]
    W[num_conv_layers+1], b[num_conv_layers+1] = self.params[(num_conv_layers+1,'W')], self.params[(num_conv_layers+1,'b')]
    
    scores = None
    reg_loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the multi-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    # Forward Pass - Stage 1 - conv - relu - 2x2 max pool                           [1]
    for i in range(num_conv_layers):
        if i is 0:

            out, conv_relu_pool_cache[i] = conv_relu_pool_forward(X, W[i], b[i], conv_params[i], pool_param)
        else:
            out, conv_relu_pool_cache[i] = conv_relu_pool_forward(out, W[i], b[i], conv_params[i], pool_param)
        reg_loss += np.sum(W[i]*W[i])  # Accumulate the Regulatization Lossed
        if verbose:
            print "CONV Layer dimension:"
            print out.shape

    # Forward Pass - Stage 2 - affine - relu                                        [2]
    out, affine_relu_cache = affine_relu_forward(out, W[num_conv_layers], b[num_conv_layers]) 
    reg_loss += np.sum(W[num_conv_layers]*W[num_conv_layers])  # Accumulate the Regulatization Lossed
    if verbose:
        print "FC Layer dimension:"
        print out.shape
    
    # Forward Pass - Stage 3 - affine                                               [3]
    scores, affine_cache = affine_forward(out, W[num_conv_layers+1], 
                                          b[num_conv_layers+1])  # Forward pass - affine
    reg_loss += np.sum(W[num_conv_layers+1]*W[num_conv_layers+1])  # Accumulate the Regulatization Lossed      
    if verbose:
        print scores.shape 
    
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
    loss, dscores = softmax_loss(scores, y)                 # Forward and Backward pass - softmax
    loss += 0.5*self.reg*reg_loss                           # Regularization - forward pass

    # Backward Pass - Stage 3 - affine                                               [3]
    dout, dW[num_conv_layers+1], db[num_conv_layers+1] = affine_backward(dscores, affine_cache)
    grads[(num_conv_layers+1,'W')] = dW[num_conv_layers+1]
    grads[(num_conv_layers+1,'b')] = db[num_conv_layers+1]
    
    # Backward Pass - Stage 2 - affine - relu                                        [2]
    dout, dW[num_conv_layers], db[num_conv_layers] = affine_relu_backward(dout, affine_relu_cache)
    grads[(num_conv_layers,'W')] = dW[num_conv_layers]
    grads[(num_conv_layers,'b')] = db[num_conv_layers]    
    
    # Backward Pass - Stage 1 - conv - relu - 2x2 max pool                           [1]
    for i in range(num_conv_layers-1,-1,-1):
        dout, dW[i], db[i] = conv_relu_pool_backward(dout, conv_relu_pool_cache[i])
        dW[i] += self.reg * W[i]                           # Regularization - backward pass 
        # Store the gradients
        grads[(i,'W')] = dW[i]   
        grads[(i,'b')] = db[i]
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
pass
