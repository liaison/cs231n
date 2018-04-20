from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def conv_relu_pool_forward_naive(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Note: use the our own naive version of CONV implementation.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward_naive(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer

    Note: use the our own naive version of CONV implementation.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db


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
                 dtype=np.float32):
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
        
        # input_dim: Tuple (C, H, W) giving size of input data
        (C, H, W) = input_dim
        
        # 1). weights for the Convolution layer
        #  w: Filter weights of shape (F, C, HH, WW)
        #  b: Biases, of shape (F,)
        
        #  F:  num_filters,  HH == WW == filter_size
        self.params['W1'] = np.random.normal(scale = weight_scale,
            size = (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        
        # 2). weights for the hidden affine layer
        # stride = 1,  padding = (F-1)/2 to ensure that
        #   the input and output of CONV layer have the same last two dimensions
        # The number of neurons in the CONV layer would be:
        #    (F, H, W) = F * H * W
        # which would then further reduced into a quarter, due to MAX pooling layer
        #    (F, H/2, W/2) = F * H * W / 4.
        # And then these neurons would serve as the input for the next fully-connected layer.
        # Therefore, the weights for this fully-connected layer should be:
        #   (num_input_neurons, num_output_neurons) = (F*H*W/4, hidden_dim)
        self.params['W2'] = np.random.normal(scale = weight_scale,
            size = (int(num_filters * H * W / 4), hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        
        # 3). weight for the output affine layer with softmax.
        #  The input of this last layer comes from the output of the previous hidden layer.
        #  And the number of output neurons is determined by the number of classes.
        # Therefore, the weights of this layer should be (hidden_dim, num_classes)
        self.params['W3'] = np.random.normal(scale = weight_scale,
            size = (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        The architecture of the network as follows:

            conv - relu - 2x2 max pool - affine - relu - affine - softmax

        The network operates on minibatches of data that have shape (N, C, H, W)
        consisting of N images, each with height H and width W and with C input
        channels.
        
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
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #     layer_1   ------------------->  layer_2 ---------> layer_3
        #  [conv - relu - 2x2 max pool] - [affine - relu] - [affine - softmax]

        # Note: one can use the provided fast version of functions in layer_utils.py
        
        out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        #out_1, cache_1 = conv_relu_pool_forward_naive(X, W1, b1, conv_param, pool_param)
        
        out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
        
        # Note: the softmax is used for the loss, not for the scores
        out_3, cache_3 = affine_forward(out_2, W3, b3)
        
        scores = out_3
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
        
        loss, dout_3 = softmax_loss(scores, y)
        # add L2 regularization
        if (self.reg):
            reg_W1 = self.reg * np.sum(np.power(self.params['W1'], 2))
            reg_W2 = self.reg * np.sum(np.power(self.params['W2'], 2))
            reg_W3 = self.reg * np.sum(np.power(self.params['W3'], 2))
            loss += 0.5 * (reg_W1 + reg_W2 + reg_W3)

        dout_2, dw3, db3 = affine_backward(dout_3, cache_3)
        dout_1, dw2, db2 = affine_relu_backward(dout_2, cache_2)
        dx, dw1, db1 = conv_relu_pool_backward(dout_1, cache_1)

        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        grads['W3'] = dw3
        grads['b3'] = db3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
