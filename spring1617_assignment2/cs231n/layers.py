from builtins import range
import numpy as np


def affine_forward(x, w, b):
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
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    # reshape the input into x' of (N, D), where D = d_1 * ... * d_k
    # then do the affine forward:  (x' . w) + b
    # the dot product (x' . w) is of shape (N, M)
    # the addition x' + b would be a broadcast addition.
    out = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])), w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: bias, of shape (M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # http://cs231n.github.io/optimization-2/
    # The "*" operator is to unzip the tuple.
    dx = dout.dot(w.T).reshape(*x.shape)
    
    # Reshape the input to (D, N) for the dot product with dout (N, M)
    # The gradient for each example is added up to be the gradients of weights.
    # As the loss = f(x1) + f(x2) + ... f(xn) where f(x1) = |x1 * w + b - y| for L1 loss.
    dw = x.reshape(x.shape[0], np.prod(x.shape[1:])).T.dot(dout)

    # Like the above two gradients, the gradient of each example adds up to the final one.
    # As the gradient is used for the loss function, which is accumulated with
    #    the loss of each example.
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x.copy()
    # set the negative output values to zero
    out[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        # the "axis" parameter is critical !!!
        sample_var = np.var(x, axis=0)
        
        # normalize the input
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        
        # scale and shift the normalized input
        out = x_normalized * gamma + beta
        
        # store the running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # prepare for the gradients calculation
        cache = (x, gamma, beta, x_normalized, sample_mean, sample_var+eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
    
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        
        out = gamma * x_normalized + beta
        
        # cache is NOT needed
        #cache = (x, gamma, beta, x_normalized, running_mean, running_var+eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    (x, gamma, beta, x_normalized, mean, var) = cache
    
    # Following the derivative from the paper: https://arxiv.org/pdf/1502.03167.pdf
    
    d_normal_x = dout * gamma

    d_var = np.sum(d_normal_x * (x - mean) * (-1.0/2) / (var ** (3.0/2)), axis=0)
    
    N = x.shape[0]
    d_mean = np.sum(-d_normal_x / np.sqrt(var), axis=0) \
            + 1.0/N * d_var * np.sum(-2*(x-mean), axis=0)
    
    dx = 1 / np.sqrt(var) * d_normal_x + d_var * 2.0 / N * (x-mean)  + 1.0/N * d_mean
    
    # the gradients of gamma and beta are easier to calculate
    dgamma = (dout * x_normalized).sum(axis=0)
    dbeta = np.sum(dout, axis=0) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    (x, gamma, beta, x_normalized, mean, var) = cache

    # Following the derivative from the paper: https://arxiv.org/pdf/1502.03167.pdf
    
    d_normal_x = dout * gamma

    d_var = np.sum(d_normal_x * (x - mean) * (-1.0/2) / (var ** (3.0/2)), axis=0)
    
    N = x.shape[0]
    # drop the second term, comparing to the previous method.
    d_mean = np.sum(-d_normal_x / np.sqrt(var), axis=0)
    
    dx = 1 / np.sqrt(var) * d_normal_x + d_var * 2.0 / N * (x-mean)  + 1.0/N * d_mean
    
    # the gradients of gamma and beta are easier to calculate
    dgamma = (dout * x_normalized).sum(axis=0)
    dbeta = np.sum(dout, axis=0) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        survival_prob = np.random.rand(*x.shape)
        # drop out the input if its survival probability 
        #    is lower than the dropout rate
        mask = survival_prob > p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # do nothing, simply pass the input to output
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
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
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def padding(x, pad):
    """
        pad the input on the last two dimensions.
        e.g.
              x of shape (N, C, H, W) 
          return: (N, C, H+pad*2, W+pad*2)
    """
    (N, C, H, W) = x.shape

    # create the expected padded ndarray with zeros
    #   and at the same pad the columns
    x_padded = np.zeros((N, C, H + pad*2, W + pad*2))
    # pad each sample
    for n in range(N):
        # iterate through each color channel, i.e. depth
        for c in range(C):
            # overwrite the padded rows with original values.
            for h in range(H):
                x_padded[n, c, h+pad, :] = np.pad(x[n, c, h, :],
                    (pad, pad), 'constant', constant_values=(0, 0))

    return x_padded


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

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
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # reference: http://cs231n.github.io/convolutional-networks/
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    (N, C, H, W) = x.shape
    # pad the input if needed
    if pad > 0:
        # create the expected padded ndarray with zeros
        x_padded = padding(x, pad)
    
    (F, C, HH, WW) = w.shape
    
    # calculate the dimensions of convolution output
    OH = int(1 + (H + 2 * pad - HH) / stride)
    OW = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, OH, OW))
    
    # convolve over element, filter/layer, height and width
    for n in range(N):
        for f in range(F):
            for ih in range(OH):
                for iw in range(OW):
                    # slice the region in 3D to convolve with weights
                    rh = ih * stride
                    rw = iw * stride
                    region = x_padded[n, :, rh:(rh+HH), rw:(rw+WW)]
                    
                    # convolve the region with the weight
                    out[n, f, ih, iw] = np.sum(region * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
        of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # unzip the cache
    # x: Input data of shape (N, C, H, W)
    # w: Filter weights of shape (F, C, HH, WW)
    # b: Biases, of shape (F,)
    (x, w, b, conv_param) = cache
    
    (N, C, H,  W)  = x.shape
    (F, C, HH, WW) = w.shape
    # dout of shape (N, F, OH, OW) where OH and OW are given by
    # OH = 1 + (H + 2 * pad - HH) / stride
    # OW = 1 + (W + 2 * pad - WW) / stride
    (N, F, OH, OW) = dout.shape
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    x_padded = padding(x, pad)

    # gradients for the padded input
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    
    # convolve over element, filter/layer, height and width
    for n in range(N):
        for f in range(F):
            for ih in range(OH):
                for iw in range(OW):
                    # slice the region in 3D to convolve with weights
                    rh = ih * stride
                    rw = iw * stride
                    region = x_padded[n, :, rh:(rh+HH), rw:(rw+WW)]
                    
                    d_error = dout[n, f, ih, iw]
                    
                    # derivative operations over the convolution.
                    #   cumulate the gradients
                    dw[f] += d_error * region
                    dx_padded[n, :, rh:(rh+HH), rw:(rw+WW)] += d_error * w[f]
    
    # retrieve the gradients from the padded input
    dx = dx_padded[:, :, pad:(H+pad), pad:(W+pad)]
    
    # Sum over all dimensions, except the filter dimension: F
    db = dout.sum(axis=3).sum(axis=2).sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']

    # out of shape (N, C, OH, OW) where OH and OW are given by
    OH = int(1 + (H - ph) / stride)
    OW = int(1 + (W - pw) / stride)
    out = np.zeros((N, C, OH, OW))

    # max over element, channel, height and width
    for n in range(N):
        for c in range(C):
            for ih in range(OH):
                for iw in range(OW):
                    # slice over the region in 2D to max over elements
                    rh = ih * stride
                    rw = iw * stride
                    region = x[n, c, rh:(rh+ph), rw:(rw+pw)]

                    # max over the region
                    out[n, c, ih, iw] = np.max(region)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
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
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # unzip the cache
    (x, pool_param) = cache

    (N, C, H, W) = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']

    # out of shape (N, C, OH, OW) where OH and OW are given by
    (N, C, OH, OW) = dout.shape
    dx = np.zeros_like(x)

    # iterate over element, channel, height and width
    # The derivative of Max operation would be passed on to the max input element
    for n in range(N):
        for c in range(C):
            for ih in range(OH):
                for iw in range(OW):
                    # slice over the region in 2D to max over elements
                    rh = ih * stride
                    rw = iw * stride
                    region = x[n, c, rh:(rh+ph), rw:(rw+pw)]

                    # get the index of the maximum input element
                    max_index = np.unravel_index(
                        np.argmax(region, axis=None), region.shape)

                    # propagate the gradient to the max input element
                    dx[n, c, rh+max_index[0], rw+max_index[1]] += dout[n, c, ih, iw]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
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
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    # transpose the dimensions of input to (C, N, H, W)
    x_transposed = np.transpose(x, (1, 0, 2, 3))
    # flatten and tranpose to (N*H*W, C)
    bn_input = x_transposed.reshape(C, -1).T

    out, cache = batchnorm_forward(bn_input, gamma, beta, bn_param)

    # convert the output back to the original dimensions
    out = np.transpose(out.T.reshape(C, N, H, W), (1, 0, 2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    (N, C, H, W) = dout.shape

    # convert the input dimensions to (N*H*W, C)
    bn_output = np.transpose(dout, (1, 0, 2, 3)).reshape(C, -1).T

    dx, dgamma, dbeta = batchnorm_backward(bn_output, cache)

    # convert the input dimension back to (N, C, H, W)
    dx = np.transpose(dx.T.reshape(C, N, H, W), (1, 0, 2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
