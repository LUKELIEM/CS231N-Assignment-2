import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

 
class DeepConvNet2(object):
  """
  A multi-layer convolutional network with following architecture:
  
  {{conv-relu}x(L - 1)-max pool}x(B - 1)-affine-relu-affine-softmax
  
  There is an arbitrary number of convolutional blocks follwed by 2 FC
  layers + Softmax.
  
  Each convolutional block consists of arbitrary number of conv-relu units
  followed by a map-pool unit, where a 2-to-1 down-sampling takes place.
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, num_filters=[[32, 32],[64, 64]], filter_sizes=[[3, 3],[3,3]], 
               input_dim=(3,32,32), hidden_dim=100, 
               num_classes=10, reg=0.0, weight_scale=1e-2, dtype=np.float32, 
               verbose=False):
    """
    Initialize a new DeepConvNet_2. 
    
    Inputs:
    The convolutional blocks and their internal conv-relu units are defined by 
    num_filters and filter_sizes:
        - num_filters: A list of lists containing the number of filters in the 
        {{conv-relu}x(L - 1)-max pool} layers
        - filter_sizes: A list of lists containing the dimensions of filters in
        the {{conv-relu}x(L - 1)-max pool} layers
    For example, num_filters=[[32, 32],[64, 64]], filter_sizes=[[5, 5],[3,3]]
    defined the architecture as followed:
    
    {CONV33x5x5-ReLU-CONV33x5x5-ReLU-MaxPool} - 
    {CONV64x3x3-ReLU-CONV64x3x3-ReLU-MaxPool} -
    {affine-relu-affine-softmax}
    
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    """
    self.reg = reg
    self.num_filters = num_filters
    self.filter_sizes = filter_sizes    
    self.dtype = dtype
    self.params = {}
    convout_dims = {}
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
    
    # Get number of CONV blocks in the architecture (B):
    # {{conv-relu}x(L - 1)-max pool}x(B - 1)-affine-relu-affine-softmax
    num_blocs = len(num_filters)
    
    # For each CONV block each containing {{conv-relu}x(L - 1)-max pool}
    for bloc in range(num_blocs):
        # Get number of CONV layers in the block (L)
        num_convs = len(num_filters[bloc])
        
        if bloc is 0:
            # In CONV Bloc1, the dimension of the input to the block is input_dim
            CC, HH, WW = input_dim
        else:
            # In subsequent CONV Blocs, the dimension of the input to the block is
            # that of the output of the maxpool in the preceding block
            CC, HH, WW = maxpool_dims[bloc-1]

            
        # With the dimension of the input defined above, we now calculate the
        # dimensions of (1) Convolution parameter W and b, and (2) the output of
        # the convolution
        for layer in range(num_convs):
            
            # First we deal with the parameters of the convolution W and b:
            # The W parameters convolve filters of dimension CxHxW on the input volume:
            #  - The number of filters is defined in num_filters
            #  - H and W are defined in filter_sizes (where H=W)
            #  - The parameter C is trickier and is described below:
            
            if layer is 0:
                # The 1st CONV layer of every block has same depth as the input
                filter_depth = CC   
            else:
                # After the 1st CONV layer, Depth = number of filters in preceding CONV layer
                filter_depth = num_filters[bloc][layer-1]
                
            # Set up weights for the filters of the CONV layer
            self.params[(bloc,layer,'W')] = weight_scale * np.random.randn(num_filters[bloc][layer], filter_depth, filter_sizes[bloc][layer], filter_sizes[bloc][layer])
            # The dimension of parameter is a vector of length = number of filters in the 
            # CONV layer
            self.params[(bloc,layer,'b')] = np.zeros(num_filters[bloc][layer])
            
            # The output of the convolution is an activation volume whereby:
            # - the depth equals the number of filters in the CONV layer
            # - the HxW is assumed to be preserved along the CONV block because of the way
            # we set up stride and padding
            convout_dims[bloc, layer] = (num_filters[bloc][layer], HH, WW)

        # The output of the last CONV layer is downsampled 2-to-1 in the maxpool layer.
        # This becomes the input to the next CONV Block
        maxpool_dims[bloc] = (num_filters[bloc][num_convs-1], HH/2, WW/2)

    # Assign weight and biases for FC layers. We treat this as a block with two FC layers.
    C, H, W = maxpool_dims[num_blocs-1]
    self.params[(num_blocs,0,'W')] = weight_scale * np.random.randn(C*H*W, hidden_dim)
    self.params[(num_blocs,0,'b')] = np.zeros(hidden_dim)
    
    self.params[(num_blocs,1,'W')] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params[(num_blocs,1,'b')] = np.zeros(num_classes)
    
    if verbose:
        # This outputs the architecture of the Deep CNN.
        print "input dimension: %d x %d x %d" % input_dim  
        print "\n"
        for bloc in range(num_blocs):
            print "CONV Block: %d" % (bloc+1)
            num_convs = len(num_filters[bloc])
            for layer in range(num_convs):
                print "  W & b in CONV layer %d" % (layer+1)
                print self.params[(bloc,layer,'W')].shape
                print self.params[(bloc,layer,'b')].shape
                print "CONV output dimension: %d x %d x %d" % convout_dims[bloc, layer]
            print "Maxpool dimension: %d x %d x %d" % maxpool_dims[bloc]  
            print "\n"
        print "W & b in FC layers:"
        print self.params[(num_blocs,0,'W')].shape
        print self.params[(num_blocs,0,'b')].shape   
        print self.params[(num_blocs,1,'W')].shape
        print self.params[(num_blocs,1,'b')].shape
             
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
    num_filters = self.num_filters
    filter_sizes = self.filter_sizes
    
    # Retrieve weights and biases for CONV layers in the CONV Blocks
    num_blocs = len(num_filters)
    for bloc in range(num_blocs):
        num_convs = len(num_filters[bloc])
        for layer in range(num_convs):
            # Retrieve weights and biases for CONV layers
            W[(bloc,layer)], b[(bloc,layer)] = self.params[(bloc,layer,'W')], self.params[(bloc,layer,'b')]
            # pass conv_param to the forward pass for the convolutional layer
            # These parameters ensures that the convolution output have the same HxW dimension
            # as the input 
            filter_size = W[(bloc,layer)].shape[2]
            conv_params[(bloc,layer)] = {'stride': 1, 'pad': (filter_size - 1) / 2}
   
        # pass pool_param to the forward pass for the max-pooling layer
        # These parameters downsamples the activation map by 2 --> H/2 x W/2
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Retrieve weights and biases for FC layers
    W[(num_blocs,0)], b[(num_blocs,0)] = self.params[(num_blocs,0,'W')], self.params[(num_blocs,0,'b')]
    W[(num_blocs,1)], b[(num_blocs,1)] = self.params[(num_blocs,1,'W')], self.params[(num_blocs,1,'b')]
    
    if verbose:
        print "Dimensions of the W parameters"
        for label, params in sorted(W.iteritems()):
            print label
            print params.shape
        print "Dimensions of the b parameters"
        for label, params in sorted(b.iteritems()):
            print label
            print params.shape
    
    scores = None
    loss, grads = 0, {}
    reg_loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the multi-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    """
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
    """
    return loss, grads
  
pass
