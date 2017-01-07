import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

 
class DeepConvNet3(object):
  """
  A multi-layer convolutional network with following architecture:
  
  {{conv-(bn)-relu}x(L)-max pool}x(B)-{affine-(bn)-relu-(DO)}-affine-softmax
  
  - There is an arbitrary number of convolutional blocks followed by 2 FC
  layers + Softmax.
  - Each convolutional block consists of arbitrary number of conv-relu units
  followed by a max-pool unit, where 2-to-1 down-sampling takes place.
  - Batch normalization is optional before each ReLU
  - Dropout is optional for the FC-ReLU layer.
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, num_filters=[[32, 32],[64, 64]], filter_sizes=[[3, 3],[3,3]], 
               input_dim=(3,32,32), hidden_dim=100, num_classes=10, xavier=False, 
               dropout=0, seed=None, use_batchnorm=False, reg=0.0, weight_scale=1e-2, 
               dtype=np.float32, verbose=False):
    """
    Initialize a new DeepConvNet3. 
    
    Inputs:
    The convolutional blocks and the conv-relu units in them are defined by the 
    two arrays num_filters and filter_sizes:
        - num_filters: An array of arrays containing the number of filters in the 
        {{conv-relu}x(L)-max pool} layers
        - filter_sizes: An array of arrays containing the dimensions of filters in
        the {{conv-relu}x(L)-max pool} layers
    For example, num_filters=[[32,32],[64,64]], filter_sizes=[[5,5],[3,3]] defines
    the architecture as followed:
    
    {CONV32x5x5-ReLU-CONV32x5x5-ReLU-MaxPool} - 
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
    self.bn_params = {}
    convout_dims = {}
    maxpool_dims = {}  
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0

    ############################################################################
    # TODO: Initialize weights and biases for the multi-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    ############################################################################
    
    # Get number of CONV blocks (B) in the architecture:
    # {{conv-relu}x(L)-max pool}x(B)-affine-relu-affine-softmax
    num_blocs = len(num_filters)
    
    # For each CONV block each containing {{conv-(bn)-relu}x(L)-max pool}
    for bloc in range(num_blocs):
        # Get number of CONV layers (L) in the block
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
                # After the 1st CONV layer, depth = number of filters in preceding CONV layer
                filter_depth = num_filters[bloc][layer-1]
                
            # Set up weights for the filters of the CONV layer
            if xavier:
                # Xavier Initialization to deal with vanishing gradient problem (encountered when L>2)
                n_input = num_filters[bloc][layer] * filter_sizes[bloc][layer] * filter_sizes[bloc][layer]
                n_output = 1.0                
                self.params[(bloc,layer,'W')] = np.sqrt(2.0 / (n_input + n_output)) * \
                            np.random.randn(num_filters[bloc][layer], filter_depth, filter_sizes[bloc][layer], \
                            filter_sizes[bloc][layer]) 
                # The dimension of b is simply a vector of length = number of filters in the 
                # CONV layer
                self.params[(bloc,layer,'b')] = np.zeros(num_filters[bloc][layer])
            else:
                self.params[(bloc,layer,'W')] = weight_scale * np.random.randn(num_filters[bloc][layer], \
                            filter_depth, filter_sizes[bloc][layer], filter_sizes[bloc][layer])
                # The dimension of b is simply a vector of length = number of filters in the 
                # CONV layer
                self.params[(bloc,layer,'b')] = np.zeros(num_filters[bloc][layer])                
                
            if self.use_batchnorm:
                self.params[(bloc,layer,'gamma')] = np.ones(num_filters[bloc][layer])
                self.params[(bloc,layer,'beta')] = np.zeros(num_filters[bloc][layer])
                self.bn_params[(bloc,layer)] = {'mode': 'train'}
            
            # The output of the convolution is an activation map volume whereby:
            # - the depth equals the number of filters in the CONV layer
            # - the HxW is assumed to be preserved along the CONV block because of the way
            # we set up stride and padding
            convout_dims[bloc, layer] = (num_filters[bloc][layer], HH, WW)

        # The output of the last CONV layer is then downsampled 2-to-1 in the maxpool layer.
        # This becomes the input to the next CONV Block
        maxpool_dims[bloc] = (num_filters[bloc][num_convs-1], HH/2, WW/2)

    # Assign weight and biases for FC layers. We treat this as a block with two FC layers.
    C, H, W = maxpool_dims[num_blocs-1]
    if xavier:
        # Xavier Initialization to deal with vanishing gradient problem (encountered when L>2)
        n_input = C*H*W
        n_output = hidden_dim
        self.params[(num_blocs,0,'W')] = np.sqrt(2.0 / (n_input + n_output)) * np.random.randn(C*H*W, hidden_dim)
        self.params[(num_blocs,0,'b')] = np.zeros(hidden_dim)
    else:
        self.params[(num_blocs,0,'W')] = weight_scale * np.random.randn(C*H*W, hidden_dim)
        self.params[(num_blocs,0,'b')] = np.zeros(hidden_dim)
        
    if self.use_batchnorm:
        self.params[(num_blocs,0,'gamma')] = np.ones(hidden_dim)
        self.params[(num_blocs,0,'beta')] = np.zeros(hidden_dim)
        self.bn_params[(num_blocs,0)] = {'mode': 'train'}
        
    self.params[(num_blocs,1,'W')] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params[(num_blocs,1,'b')] = np.zeros(num_classes)        
    
    if verbose:
        print "This outlines the architecture of the Deep CNN:"
        print "input dimension: %d x %d x %d" % input_dim  
        print "\n"
        for bloc in range(num_blocs):
            print "CONV Block: %d" % (bloc+1)
            num_convs = len(num_filters[bloc])
            for layer in range(num_convs):
                print "  W & b in CONV layer %d" % (layer+1)
                print self.params[(bloc,layer,'W')].shape
                print self.params[(bloc,layer,'b')].shape
                if self.use_batchnorm:
                    print "Gamma and Beta in CONV-ReLU layers:"
                    print self.params[(bloc,layer,'gamma')].shape
                    print self.params[(bloc,layer,'beta')].shape                 
                print "CONV output dimension: %d x %d x %d" % convout_dims[bloc, layer]
            print "Maxpool dimension: %d x %d x %d" % maxpool_dims[bloc]  
            print "\n"
        print "W & b in FC layers:"
        print self.params[(num_blocs,0,'W')].shape
        print self.params[(num_blocs,0,'b')].shape   
        if self.use_batchnorm:
            print "Gamma and Beta in FC layers:"
            print self.params[(num_blocs,0,'gamma')].shape
            print self.params[(num_blocs,0,'beta')].shape         
        print self.params[(num_blocs,1,'W')].shape
        print self.params[(num_blocs,1,'b')].shape
        print "\n"
        if self.use_batchnorm:
            print "BN parameters for CONV and FC1:"
            print self.bn_params
            
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    if verbose:
        print "dropout parameters:"
        print self.dropout_param           
        
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    

  def loss(self, X, y=None, verbose=False, debug=False):
    """
    Evaluate loss and gradient for the multi-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    W = {}
    b = {}
    gamma={}
    beta={}
    dW = {}
    db = {}
    dgamma={}
    dbeta={}
    conv_params = {}
    conv_relu_cache = {}
    conv_relu_pool_cache = {}
    conv_bn_relu_cache = {}
    conv_bn_relu_pool_cache = {}
    pool_out={}
    num_filters = self.num_filters
    filter_sizes = self.filter_sizes
    
    mode = 'test' if y is None else 'train'
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for label, bn_param in self.bn_params.iteritems():
          bn_param['mode'] = mode
    
    # Retrieve weights and biases for CONV layers in the CONV Blocks
    num_blocs = len(num_filters)
    for bloc in range(num_blocs):
        num_convs = len(num_filters[bloc])
        for layer in range(num_convs):
            # Retrieve weights and biases for CONV layers
            W[(bloc,layer)], b[(bloc,layer)] = self.params[(bloc,layer,'W')], self.params[(bloc,layer,'b')]
            
            if self.use_batchnorm:
                # Retrieve gammas and betas for CONV layers
                gamma[(bloc,layer)], beta[(bloc,layer)] = self.params[(bloc,layer,'gamma')], self.params[(bloc,layer,'beta')]
                
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
    if self.use_batchnorm:
        # Retrieve gammas and betas for the FC layers
        gamma[(num_blocs,0)], beta[(num_blocs,0)] = self.params[(num_blocs,0,'gamma')], self.params[(num_blocs,0,'beta')]    
    W[(num_blocs,1)], b[(num_blocs,1)] = self.params[(num_blocs,1,'W')], self.params[(num_blocs,1,'b')]
    
    scores = None
    reg_loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the multi-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    # Forward Pass - Stage 1 - {{conv-(bn)-relu}x(L)-max pool}x(B)           [1]
    num_blocs = len(num_filters)
    if verbose:
        print "Running Forward Pass throught the DNN:"
    
    ############################################################################
    # There are two major branches in our code here related to batch norm:     #
    #   - The first branch places a batch norm layer before every ReLU         #
    ############################################################################    
    if self.use_batchnorm:
        # When batch norm is turned on
        for bloc in range(num_blocs):
            num_convs = len(num_filters[bloc])
            if bloc is 0:
                bloc_in = X
            else:
                bloc_in = pool_out[bloc-1]
            
            for layer in range(num_convs):
                if layer is 0:
                    if num_convs is 1:
                    # If there is only 1 layer in the block
                        pool_out[bloc], conv_bn_relu_pool_cache[bloc] = conv_bn_relu_pool_forward(bloc_in, W[(bloc,layer)], 
                                            b[(bloc,layer)], gamma[(bloc,layer)], beta[(bloc,layer)], 
                                            conv_params[(bloc,layer)], pool_param, self.bn_params[(bloc,layer)])
                        if verbose:
                            print "CONV Block %d - Layer %d:" % (bloc,layer)
                            print "Conv-bn-reLU-pool forward"                    
                            print pool_out[bloc].shape
                            if debug:
                                print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))
                    else: 
                    # If this is the first layer in the block
                        out, conv_bn_relu_cache[(bloc,layer)] = conv_bn_relu_forward(bloc_in, W[(bloc,layer)], 
                                            b[(bloc,layer)], gamma[(bloc,layer)], beta[(bloc,layer)], 
                                            conv_params[(bloc,layer)], self.bn_params[(bloc,layer)])
                        if verbose:
                            print "CONV Block %d - Layer %d:" % (bloc,layer)
                            print "Conv-bn-reLU forward"                    
                            print out.shape
                            if debug:
                                print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))                        
                elif (layer is (num_convs-1)):
                    # If this is the last layer in the block
                    pool_out[bloc], conv_bn_relu_pool_cache[bloc] = conv_bn_relu_pool_forward(out, W[(bloc,layer)], 
                                            b[(bloc,layer)], gamma[(bloc,layer)], beta[(bloc,layer)], 
                                            conv_params[(bloc,layer)], pool_param, self.bn_params[(bloc,layer)])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-bn-reLU-pool forward"                    
                        print pool_out[bloc].shape
                        if debug:
                            print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))                            
                else:
                    # For all the CONV layers between the 1st and the last
                    out, conv_bn_relu_cache[(bloc,layer)] = conv_bn_relu_forward(out, W[(bloc,layer)], \
                                             b[(bloc,layer)], gamma[(bloc,layer)], beta[(bloc,layer)], \
                                             conv_params[(bloc,layer)], self.bn_params[(bloc,layer)])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-bn-reLU forward"
                        print out.shape
                        if debug:
                            print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))
                    
                reg_loss += np.sum(W[(bloc,layer)]*W[(bloc,layer)])  # Accumulate the Regulatization Losses
    ############################################################################
    # The second branch simply omits the batch norm layer.                     #
    ############################################################################        
    else:
        # When batch norm is turned off
        for bloc in range(num_blocs):
            num_convs = len(num_filters[bloc])
            if bloc is 0:
                bloc_in = X
            else:
                bloc_in = pool_out[bloc-1]
            
            for layer in range(num_convs):
                if layer is 0:
                    if num_convs is 1:
                        # When there is only 1 CONV layer in the CONV block
                        pool_out[bloc], conv_relu_pool_cache[bloc] = conv_relu_pool_forward(bloc_in, 
                                                                                        W[(bloc,layer)], b[(bloc,layer)],
                                                                                        conv_params[(bloc,layer)], pool_param)
                        if verbose:
                            print "CONV Block %d - Layer %d:" % (bloc,layer)
                            print "Conv-reLU-pool forward"                    
                            print pool_out[bloc].shape
                            if debug:
                                print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))
                    else:
                        # When there is more than 1 CONV layer in the CONV block
                        out, conv_relu_cache[(bloc,layer)] = conv_relu_forward(bloc_in, W[(bloc,layer)], 
                                                                           b[(bloc,layer)], conv_params[(bloc,layer)])
                        if verbose:
                            print "CONV Block %d - Layer %d:" % (bloc,layer)
                            print "Conv-reLU forward"
                            print out.shape
                            if debug:
                                print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))

                elif layer is (num_convs-1):
                    # When this is the last CONV layer in the CONV block
                    pool_out[bloc], conv_relu_pool_cache[bloc] = conv_relu_pool_forward(out, W[(bloc,layer)], b[(bloc,layer)],
                                                                                    conv_params[(bloc,layer)], pool_param)
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-reLU-pool forward"                    
                        print pool_out[bloc].shape
                        if debug:
                            print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))
                else:
                    # For all the CONV layers between the 1st and the last
                    out, conv_relu_cache[(bloc,layer)] = conv_relu_forward(out, W[(bloc,layer)], b[(bloc,layer)], conv_params[(bloc,layer)])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-reLU forward"
                        print out.shape
                        if debug:
                            print "CONV output mean: %f  std: %f" % (np.mean(out), np.std(out))
                    
                reg_loss += np.sum(W[(bloc,layer)]*W[(bloc,layer)])  # Accumulate the Regulatization Losses

    ############################################################################
    # At this point, we are done with the forward pass of the convolutional    #
    # blocks defined by {{conv-(bn)-relu}x(L)-max pool}x(B). We next move onto #
    # the 2 FC layers.                                                         #
    ############################################################################        
                
    # Forward Pass - Stage 2 - affine - relu                                        [2]
    if self.use_batchnorm:
        out, affine_bn_relu_cache = affine_bn_relu_forward(pool_out[num_blocs-1], W[(num_blocs,0)], b[(num_blocs,0)],
                                    gamma[(num_blocs,0)], beta[(num_blocs,0)], self.bn_params[(num_blocs,0)])        
    else:
        out, affine_relu_cache = affine_relu_forward(pool_out[num_blocs-1], W[(num_blocs,0)], b[(num_blocs,0)]) 
    reg_loss += np.sum(W[(num_blocs,0)]*W[(num_blocs,0)])  # Accumulate the Regulatization Lossed
    if verbose:
        print "FC Layer 1 output dimension:"
        print out.shape
        if debug:
            print "FC Layer output mean: %f  std: %f" % (np.mean(out), np.std(out))    
            
    # Forward Pass - Stage 3 - affine                                               [3]
    scores, affine_cache = affine_forward(out, W[(num_blocs,1)], b[(num_blocs,1)])  # Forward pass - affine
    reg_loss += np.sum(W[(num_blocs,1)]*W[(num_blocs,1)])  # Accumulate the Regulatization Lossed      
    if verbose:
        print "FC Layer 2 output dimension:"
        print scores.shape 
    
    ############################################################################
    #                             END OF FORWARD PASS CODE                     #
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

    if verbose:
        print "Running Backward Pass throught the DNN:"

    # Backward Pass - Stage 3 - affine                                               [3]
    dout, dW[(num_blocs,1)], db[(num_blocs,1)] = affine_backward(dscores, affine_cache)
    dW[(num_blocs,1)] += self.reg * W[(num_blocs,1)]   # Regularization - backward pass 
    grads[(num_blocs,1,'W')] = dW[(num_blocs,1)]
    grads[(num_blocs,1,'b')] = db[(num_blocs,1)]

    if self.use_batchnorm:  
        # If batch normalization is turned on
        if verbose:
            print "FC Layer 2 dout dimension:"
            print dout.shape  
        
        # Backward Pass - Stage 2 - affine-bn-relu                                   [2]
        dout,dW[(num_blocs,0)],db[(num_blocs,0)],dgamma[(num_blocs,0)],dbeta[(num_blocs,0)] = affine_bn_relu_backward(dout,
                                        affine_bn_relu_cache)
        dW[(num_blocs,0)] += self.reg * W[(num_blocs,0)]   # Regularization - backward pass 
        grads[(num_blocs,0,'W')] = dW[(num_blocs,0)]
        grads[(num_blocs,0,'b')] = db[(num_blocs,0)]
        grads[(num_blocs,0,'gamma')] = dgamma[(num_blocs,0)]
        grads[(num_blocs,0,'beta')] = dbeta[(num_blocs,0)]
        
        if verbose:
            print "FC Layer 1 dout dimension:"
            print dout.shape 
        
        # Backward Pass - Stage 1 - {{conv-bn-relu}x(L)-max pool}x(B)                [1]
        for bloc in range(num_blocs-1,-1,-1):
            num_convs = len(num_filters[bloc])
            for layer in range(num_convs-1,-1,-1):        
                if layer is (num_convs-1):
                    dout, dW[(bloc,layer)], db[(bloc,layer)],dgamma[(bloc,layer)],dbeta[(bloc,layer)] = conv_bn_relu_pool_backward(dout, 
                                                                                  conv_bn_relu_pool_cache[bloc])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-bn-reLU-pool backward"
                        print dout.shape
                else:
                    dout, dW[(bloc,layer)], db[(bloc,layer)],dgamma[(bloc,layer)],dbeta[(bloc,layer)] = conv_bn_relu_backward(dout, 
                                                                                  conv_bn_relu_cache[(bloc,layer)])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-bn-reLU backward"
                        print dout.shape
                    
                dW[(bloc,layer)] += self.reg * W[(bloc,layer)]   # Regularization - backward pass 
            
                # Store the gradients
                grads[(bloc,layer,'W')] = dW[(bloc,layer)]   
                grads[(bloc,layer,'b')] = db[(bloc,layer)] 
                grads[(bloc,layer,'gamma')] = dgamma[(bloc,layer)]
                grads[(bloc,layer,'beta')] = dbeta[(bloc,layer)]                
    else:
        # If batch normalization is turned off
        if verbose:
            print "FC Layer 2 dout dimension:"
            print dout.shape  
        
        # Backward Pass - Stage 2 - affine - relu                                        [2]
        dout, dW[(num_blocs,0)], db[(num_blocs,0)] = affine_relu_backward(dout, affine_relu_cache)
        dW[(num_blocs,0)] += self.reg * W[(num_blocs,0)]   # Regularization - backward pass 
        grads[(num_blocs,0,'W')] = dW[(num_blocs,0)]
        grads[(num_blocs,0,'b')] = db[(num_blocs,0)]   

        if verbose:
            print "FC Layer 1 dout dimension:"
            print dout.shape 
        
        # Backward Pass - Stage 1 - {{conv-relu}x(L)-max pool}x(B)                       [1]
        for bloc in range(num_blocs-1,-1,-1):
            num_convs = len(num_filters[bloc])
            for layer in range(num_convs-1,-1,-1):        
                if layer is (num_convs-1):
                    dout, dW[(bloc,layer)], db[(bloc,layer)] = conv_relu_pool_backward(dout, 
                                                                                  conv_relu_pool_cache[bloc])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-reLU-pool backward"
                        print dout.shape
                else:
                    dout, dW[(bloc,layer)], db[(bloc,layer)] = conv_relu_backward(dout, 
                                                                             conv_relu_cache[(bloc,layer)])
                    if verbose:
                        print "CONV Block %d - Layer %d:" % (bloc,layer)
                        print "Conv-reLU backward"
                        print dout.shape
                    
                dW[(bloc,layer)] += self.reg * W[(bloc,layer)]   # Regularization - backward pass 
            
                # Store the gradients
                grads[(bloc,layer,'W')] = dW[(bloc,layer)]   
                grads[(bloc,layer,'b')] = db[(bloc,layer)]
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
  
pass
