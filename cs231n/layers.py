import numpy as np
from numpy import unravel_index

def affine_forward(x, w, b, verbose=False):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################

  X_vector = np.reshape(x, (x.shape[0], -1))  # reshape x into a vector of dimension D
   
  out = np.dot(X_vector,w)+b.reshape(1,-1)    # forward pass - dot(w,x)+b
    
  if verbose:  
      print out.shape
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache, verbose=False):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  X_vector = np.reshape(x, (x.shape[0], -1))  # reshape x into a vector of dimension D
  
  dX_vector = np.dot(dout, w.T)
  dx = dX_vector.reshape(*x.shape)
  dw = np.dot(X_vector.T, dout)
  db = np.sum(dout, axis=0)  

  if verbose:  
      print X_vector.shape
      print dx.shape
      print dout.shape
      print dw.shape
      print db.shape
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################

  out = np.maximum(0,x)  # This is ReLU
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  
  # The local gradient for the ReLU is effectively a switch:
  #     Since r=max(0,x), we have that dr/dx=1(x>0).
  # (http://cs231n.github.io/neural-networks-case-study/)
  # The ReLU unit lets the gradient pass through unchanged if its input 
  # was greater than 0, but kills it if its input was less than zero 
  # during the forward pass.
  
  dx = dout
  dx[x<=0]=0
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param, verbose=False):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    # Stage 1 - # x - mean(x) --> sub                            [1]
    mean = np.mean(x, axis=0)
    sub = x - mean
    
    # Stage 2 - sqrt(1/m * sum(sub*sub) --> stdev                [2]    
    sub_sq = sub*sub
    var = np.mean(sub_sq, axis=0)
    stdev = np.sqrt(var + eps)
    
    dstdev_dvar = 1.0/(2.0 * stdev)
        
    # Stage 3 - sub/stdev --> norm_x                             [3]    
    norm_x = sub/stdev

    dnormx_dstdev = -1.0 * sub/(stdev*stdev)
    
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * var
    
    if verbose:
        print "Stage 2 local gradients"
        print var
        print stdev
        print dstdev_dvar
        print "Running means and vars"
        print running_mean
        print running_var

    # Stage 4 - norm_x*gamma + beta --> out                      [4]      
    out = gamma * norm_x + beta
    dout_dgamma = norm_x

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    if verbose:
        print running_mean
        print running_var
       
    out  = gamma * (x - running_mean)/np.sqrt(running_var) + beta
    
    # During test mode, zero out all variables --> cache
    sub = 0
    stdev = 0
    dstdev_dvar = 0
    dnormx_dstdev = 0
    dout_dgamma = 0
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  
  cache = (x, N, gamma, sub, stdev, dstdev_dvar, dnormx_dstdev, dout_dgamma)

  return out, cache


def batchnorm_backward(dout, cache, verbose=False):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
    
  x, N, gamma, sub, stdev, dstdev_dvar, dnormx_dstdev, dout_dgamma = cache  
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  # Stage 4 - Backprop gamma*norm_x + beta --> out                      [4]      
  dbeta = np.sum(dout, axis=0, keepdims=True)
  dgamma =  np.sum(dout_dgamma*dout, axis=0, keepdims=True)
  dnorm_x = gamma * dout
 
  # Stage 3 - Backprop sub/stdev --> norm_x                             [3]      
  dstdev = np.sum(dnormx_dstdev*dnorm_x, axis=0, keepdims=True)
  dsub = 1.0/stdev * dnorm_x

  # Stage 2 - # sqrt(1/m * sum(sub*sub) --> stdev                       [2]    
  dvar = dstdev_dvar * dstdev 
  dsub_sq = np.zeros_like(x) + dvar/N
  dsub += (2.0 * sub * dsub_sq)

  # Stage 1 - # x - 1/m*sum(x) --> sub                                  [1]
  dx = dsub
  dmean = np.sum(dsub, axis=0, keepdims=True)
  dx += -1.0/N * dmean  
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  if verbose: 
    print "Stage 2 Backprop:"
    print dstdev_dvar
    print dstdev
    print dvar
    print dsub_sq
    print dsub
    print dmean
    print dx

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
    
  x, N, gamma, sub, stdev, dstdev_dvar, dnormx_dstdev, dout_dgamma = cache 
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################

  # Stage 2 - Backprop gamma*norm_x + beta --> out     
  dbeta = np.sum(dout, axis=0, keepdims=True)
  dgamma =  np.sum(dout_dgamma*dout, axis=0, keepdims=True)
  dnorm_x = gamma * dout

  # Stage 1 - Backprop (x-mean)/stdev --> norm_x
  # The equation below is derived by taking the partial derivative of norm_x wrt sub
  dsub =  1.0/stdev * dnorm_x - (sub * np.mean(sub/stdev**3 * dnorm_x, axis=0, keepdims=True))
  dmean = np.mean(dsub, axis=0, keepdims=True)
  dx = dsub - dmean  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param, verbose=False):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    mask = (np.random.rand(*x.shape) < (1-p)) / (1-p)   # create dropout mask
    out = x * mask     # drop
    
    if verbose:
        print "Input:"        
        print x
        print "Mask:"           
        print mask
        print "Output:" 
        print out
          
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache, verbose=False):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = mask * dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param, verbose=False):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
    
  # Calculate dimension of activation map layer
  H_Act = 1 + (H + 2 * pad - HH) / stride
  W_Act = 1 + (W + 2 * pad - WW) / stride
  out_shape = (N, F, H_Act, W_Act)
  out = np.zeros(out_shape)
  
  # Pad the input volumes
  npad = ((0, 0), (0, 0), (1, 1), (1, 1))
  padded_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

  
  # A brute force way to implement forward pass for convolution
  for i in range(N):   # for each padded image
      for f in range(F):    # for each filter
          for j in range(H_Act):   # Take stride horizontally in image
              for k in range(W_Act):    # Take stride vertically in image
                  # There is only 1 stage in the forward pass
                  # - Dot product the filter with the associated sub-volume in the image
                  out[i,f,j,k] = np.sum(padded_x[i,:,j*stride:j*stride+HH, k*stride:k*stride+WW]*w[f])+b[f]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
    
  # Pad the input volumes
  npad = ((0, 0), (0, 0), (1, 1), (1, 1))
  padded_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  
  N, F, H_out, W_out = dout.shape

  dpadded_x = np.zeros(padded_x.shape)   
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # Backward pass for the convolution
  for i in range(N):   # for each padded image
      for f in range(F):    # for each filter
          for j in range(H_out):   # Take stride horizontally in image
              for k in range(W_out):    # Take stride vertically in image
                  # There is only 1 stage in the forward pass
                  # - Backprop the convolution equation:
                  # out[i,f,j,k] = np.sum(padded_x[i,:,j*stride:j*stride+HH, k*stride:k*stride+WW]*w[f])
                  #                +b[f]
                  
                  dpadded_x[i,:,j*stride:j*stride+HH, k*stride:k*stride+WW] += w[f] * dout[i,f,j,k]
                  dw[f] += padded_x[i,:,j*stride:j*stride+HH, k*stride:k*stride+WW] * dout[i,f,j,k]
                  db[f] += dout[i,f,j,k]

  # unpad the padded volume
  dx = dpadded_x[:,:,1:-1,1:-1]
  
  # This is a more efficient way to get db
  # db = np.sum(dout,axis=(0,2,3))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
    
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  # Calculate dimension of pool map
  HH = 1 + (H - pool_height) / stride
  WW = 1 + (W - pool_width) / stride
  
  out_shape = (N, C, HH, WW)
  out = np.zeros(out_shape)
  max_h = np.zeros(out_shape)
  max_w = np.zeros(out_shape)

  for n in range(N):   # for each image
      for c in range(C):    # for each channel 
          for j in range(HH):   # Take stride horizontally in image
              for k in range(WW):    # Take stride vertically in image
                  # Forward Pass - There is only 1 Stage
                  # Perform the max pooling operation
                  pool = x[n,c,j*stride:j*stride+pool_height, k*stride:k*stride+pool_width] 
                  out[n,c,j,k] = np.max(pool)
                    
                  # These stores the location of the max - for backprop
                  h, w = np.unravel_index(pool.argmax(), pool.shape)
                  max_h[n,c,j,k] = h
                  max_w[n,c,j,k] = w

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, max_h, max_w, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, max_h, max_w, pool_param = cache
  N, C, H, W = x.shape
  N, C, HH, WW = dout.shape
  dx = np.zeros(x.shape)
    
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  for n in range(N):   # for each image
      for c in range(C):    # for each channel 
          for j in range(HH):   # Take stride horizontally in image
              for k in range(WW):    # Take stride vertically in image
                  # Backward the max pooling operation
                  # dout is routed through the max location of the pooling mask
                  dx[n,c,j*stride+max_h[n,c,j,k], k*stride+max_w[n,c,j,k]] +=  dout[n,c,j,k]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param, verbose=False):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape
  running_mean = bn_param.get('running_mean', np.zeros((C, H, W), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((C, H, W), dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
    # Stage 1 - # x - mean(x) --> sub                            [1]
    mean = np.mean(x, axis=(0, 2, 3))
    mean = np.reshape(mean,(C,1,1))
    sub = x - mean
    
    # Stage 2 - sqrt(1/m * sum(sub*sub) --> stdev                [2]    
    stdev = x.std(axis=(0, 2, 3))
    stdev = np.reshape(stdev,(C,1,1))
    dstdev_dvar= 1.0/(2.0 * stdev)
        
    # Stage 3 - sub/stdev --> norm_x                             [3]    
   
    norm_x = sub/stdev
    var = stdev*stdev
    dnormx_dstdev = -1.0 * sub/var
    
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * var
    
    if verbose:
        print "Stage 2 local gradients"
        print mean.shape
        print stdev.shape
        # print dstdev_dvar
        # print "Running means and vars"
        # print running_mean
        # print running_var

    # Stage 4 - norm_x*gamma + beta --> out                      [4]
    out = np.reshape(gamma, (C,1,1)) * norm_x + np.reshape(beta, (C,1,1))
    dout_dgamma = norm_x
    
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    if verbose:
        print running_mean
        print running_var
       
    out  = np.reshape(gamma, (C,1,1)) * (x - running_mean)/np.sqrt(running_var) + np.reshape(beta, (C,1,1))
    
    # During test mode, zero out all variables --> cache
    sub = 0
    stdev = 0
    dstdev_dvar = 0
    dnormx_dstdev = 0
    dout_dgamma = 0
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  
  cache = (x, gamma, sub, stdev, dout_dgamma)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  x, gamma, sub, stdev, dout_dgamma = cache 
  N, C, H, W = x.shape
  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  # Stage 2 - Backprop gamma*norm_x + beta --> out                      [2]      
  dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
  dbeta = np.reshape(dbeta, (1,-1))   # Reshape into vector
  dgamma =  np.sum(dout_dgamma*dout, axis=(0, 2, 3), keepdims=True)
  dgamma = np.reshape(dgamma, (1,-1))  # Reshape into vector
  dnorm_x = np.reshape(gamma, (C,1,1)) * dout

  # Stage 1 - Backprop (x-mean)/stdev --> norm_x                        [1]
  # The equation below is derived by taking the partial derivative of norm_x wrt sub
  dsub =  1.0/stdev * dnorm_x - (sub/stdev**3 * np.mean(sub * dnorm_x, axis=(0, 2, 3), keepdims=True))
  dmean = np.mean(dsub, axis=(0, 2, 3), keepdims=True)
  dx = dsub - dmean 

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
