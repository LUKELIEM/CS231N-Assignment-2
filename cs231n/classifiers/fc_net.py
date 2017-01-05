import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0, verbose=False):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    if verbose:
        print self.params
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    hidden_layer, cache1 = affine_relu_forward(X, W1, b1)  # Forward pass - affine-ReLU
    scores, cache2 = affine_forward(hidden_layer, W2, b2)  # Forward pass - affine
  
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    loss, dscore = softmax_loss(scores, y)                  # Forward and Backward pass - softmax
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))    # Regularization - forward pass
    
    dhidden, dW2, db2 = affine_backward(dscore, cache2)     # Backward pass - affine
    dx, dW1, db1 = affine_relu_backward(dhidden, cache1)    # Forward pass - affine-ReLU
    
    # Regularization - backward pass
    dW1 += self.reg * W1 
    dW2 += self.reg * W2
    
    # Store into grads dictionary
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None, verbose=False):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
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
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    num_layers = self.num_layers
    
    for layer in range(num_layers):
        if layer is 0:
            self.params[(0,'W')] = weight_scale * np.random.randn(input_dim, hidden_dims[layer])
            self.params[(0,'b')] = np.zeros(hidden_dims[layer])            
            if self.use_batchnorm:
                self.params[(0,'gamma')] = np.ones(hidden_dims[layer])
                self.params[(0,'beta')] = np.zeros(hidden_dims[layer])            
            
        elif layer is (num_layers-1):
            self.params[(layer,'W')] = weight_scale * np.random.randn(hidden_dims[layer-1], num_classes)
            self.params[(layer,'b')] = np.zeros(num_classes)
            # Note that there is no batch normalization in the final layer before softmax
        else:
            self.params[(layer,'W')] = weight_scale * np.random.randn(hidden_dims[layer-1], hidden_dims[layer])
            self.params[(layer,'b')] = np.zeros(hidden_dims[layer])
            if self.use_batchnorm:
                self.params[(layer, 'gamma')] = np.ones(hidden_dims[layer])
                self.params[(layer, 'beta')] = np.zeros(hidden_dims[layer])            
  
    if verbose:
        for layer in range(num_layers):
            print "W & b in layer %d" % layer
            print self.params[(layer,'W')].shape
            print self.params[(layer,'b')].shape
        if self.use_batchnorm:    
            for layer in range(num_layers-1):
                print "BN gamma and beta in layer %d" % layer
                print self.params[(layer,'gamma')].shape
                print self.params[(layer,'beta')].shape

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
        if verbose:
            print "BN parameters:"
            print self.bn_params    
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None, verbose=False):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    W = {}
    b = {}
    gamma = {}
    beta = {}
    hidden = {}
    cache = {}
    dW = {}
    db = {}
    dgamma = {}
    dbeta = {}
    dhidden = {}
    dropout_cache={}
    
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    
    if self.use_batchnorm:
      for bn_param in self.bn_params:
          bn_param['mode'] = mode

    scores = None
    reg_loss = 0
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    num_layers = self.num_layers
    
    for i in range(num_layers):
        # Retrieve weights and biases
        W[i], b[i] = self.params[(i,'W')], self.params[(i,'b')]
        
        if self.use_batchnorm:
            # Batch normalization is on
            if i is 0:
                gamma[i], beta[i] = self.params[(0,'gamma')], self.params[(0,'beta')]
                hidden[i], cache[i] = affine_bn_relu_forward(X, W[i], 
                                                             b[i], gamma[i], beta[i], 
                                                             self.bn_params[0])     # Forward pass - affine-bn-ReLU
                if self.use_dropout:
                    hidden[i], dropout_cache[i] = dropout_forward(hidden[i], 
                                                                 self.dropout_param)      # optional dropout
                    
            elif i is (num_layers-1):
                scores, cache[i] = affine_forward(hidden[i-1], W[i], b[i])          # Forward pass - affine
            else:
                gamma[i], beta[i] = self.params[(i,'gamma')], self.params[(i,'beta')]
                hidden[i], cache[i] = affine_bn_relu_forward(hidden[i-1], 
                                                             W[i], b[i], gamma[i], beta[i], 
                                                             self.bn_params[i])     # Forward pass - affine-bn-ReLU
                if self.use_dropout:
                    hidden[i], dropout_cache[i] = dropout_forward(hidden[i], 
                                                                 self.dropout_param)      # optional dropout 
        else:
            # Batch normalization is off
            if i is 0:
                out, cache[i] = affine_relu_forward(X, W[i], b[i])    # Forward pass - affine-ReLU
                if self.use_dropout:
                    hidden[i], dropout_cache[i] = dropout_forward(out, 
                                                                 self.dropout_param, verbose=verbose)      # optional dropout    
                else:
                    hidden[i] = out
            elif i is (num_layers-1):
                scores, cache[i] = affine_forward(hidden[i-1], W[i], b[i])  # Forward pass - affine
            else:
                out, cache[i] = affine_relu_forward(hidden[i-1],
                                                          W[i], b[i])       # Forward pass - affine-ReLU
                if self.use_dropout:
                    hidden[i], dropout_cache[i] = dropout_forward(out, 
                                                                 self.dropout_param, verbose=verbose)      # optional dropout    
                else:
                    hidden[i] = out
                    
        reg_loss += np.sum(W[i]*W[i])  # Accumulate the Regulatization Lossed
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    loss, dscore = softmax_loss(scores, y)                  # Forward and Backward pass - softmax
    loss += 0.5*self.reg*reg_loss                           # Regularization - forward pass

    for i in range(num_layers-1,-1,-1):
        if self.use_batchnorm:
            # Batch normalization is on
            if i is (num_layers-1):
                dhidden[i], dW[i], db[i] = affine_backward(dscore, cache[i]) # Backward pass - affine
            elif i is 0:
                if self.use_dropout:
                    dhidden[i+1] = dropout_backward(dhidden[i+1], dropout_cache[i])      # optional dropout 
                __, dW[i], db[i], dgamma[i], dbeta[i] = affine_bn_relu_backward(dhidden[i+1], 
                                                        cache[i])            # Backward pass - affine-bn-ReLU
                # Store the gradients for batch normalization
                grads[(i,'gamma')] = dgamma[i]   
                grads[(i,'beta')] = dbeta[i]
            else:
                if self.use_dropout:
                    dhidden[i+1] = dropout_backward(dhidden[i+1], dropout_cache[i])      # optional dropout 
                dhidden[i], dW[i], db[i], dgamma[i], dbeta[i] = affine_bn_relu_backward(dhidden[i+1], 
                                                                                        cache[i])    # Backward pass - affine-bn-ReLU
                # Store the gradients for batch normalization
                grads[(i,'gamma')] = dgamma[i]   
                grads[(i,'beta')] = dbeta[i]
        else: 
            # Batch normalization is off
            if i is (num_layers-1):
                dhidden[i], dW[i], db[i] = affine_backward(dscore, cache[i]) # Backward pass - affine
            elif i is 0:
                if self.use_dropout:
                    dhidden[i+1] = dropout_backward(dhidden[i+1], dropout_cache[i])      # optional dropout 
                __, dW[i], db[i] = affine_relu_backward(dhidden[i+1], 
                                                        cache[i])            # Backward pass - affine-ReLU
            else:
                if self.use_dropout:
                    dhidden[i+1] = dropout_backward(dhidden[i+1], dropout_cache[i])      # optional dropout 
                dhidden[i], dW[i], db[i] = affine_relu_backward(dhidden[i+1], 
                                                                cache[i])    # Backward pass - affine-ReLU
            
        dW[i] += self.reg * W[i]                           # Regularization - backward pass 

        # Store the gradients
        grads[(i,'W')] = dW[i]   
        grads[(i,'b')] = db[i]   
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
