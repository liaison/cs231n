from builtins import range
from builtins import object
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
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
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
        self.params['W1'] = \
            np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = \
            np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # The data flow for this two-layer network is [affine - relu - affine - softmax].
        out_1, cache_1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out_2, cache_2 = affine_forward(out_1, self.params['W2'], self.params['b2'])
        # Note: NO softmax on the final output
        scores = out_2

        # the following code adds the softmax layer on the score values.
        #exp_out_2 = np.exp(out_2)
        #scores = (1 / np.sum(exp_out_2, axis=1)).reshape(-1, 1) * exp_out_2
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
        loss, d_out_2 = softmax_loss(out_2, y)

        # add L2 regularization
        if (self.reg):
            reg_W1 = self.reg * np.sum(np.power(self.params['W1'], 2))
            reg_W2 = self.reg * np.sum(np.power(self.params['W2'], 2))
            loss += 0.5 * self.reg * (reg_W1 + reg_W2)

        d_out_1, dw2, db2 = affine_backward(d_out_2, cache_2)
        dx, dw1, db1 = affine_relu_backward(d_out_1, cache_1)

        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a
      Batch Normalization and then the ReLU gate

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    fc_out, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache

    d_relu = relu_backward(dout, relu_cache)
    d_bn, dgamma, dbeta = batchnorm_backward(d_relu, bn_cache)
    dx, dw, db = affine_backward(d_bn, fc_cache)
    
    return dx, dw, db, dgamma, dbeta


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
                 weight_scale=1e-2, dtype=np.float32, seed=None):
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

        # construct a chain to determine the dimension for each layer
        dimension_chain = [input_dim]
        dimension_chain.extend(hidden_dims)
        dimension_chain.append(num_classes)

        for i in range(0, len(hidden_dims) + 1):
            # init the weights for hidden layers and output layer
            self.params['W' + str(i+1)] = \
                np.random.normal(scale = weight_scale,
                                 size = (dimension_chain[i], dimension_chain[i+1]))
            self.params['b' + str(i+1)] = np.zeros(dimension_chain[i+1])

            # init the scale and shift parameters for Batch Normalization
            if self.use_batchnorm and i != len(hidden_dims):
                self.params['gamma' + str(i+1)] = np.ones(dimension_chain[i+1])
                self.params['beta' + str(i+1)] = np.zeros(dimension_chain[i+1])

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
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def get_weights(self):
        """
        Return the dictionary of weights and bias for the hidden layers and the output layer
        """
        return self.params


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
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
        
        out_cache_dict = {}
        
        layer_input = X
        # iterate over the hidden layers and the final output layer
        for i in range(self.num_layers):
            layer_index = str(i+1)

            if (i == self.num_layers - 1):
                # the final output layer would NOT need the ReLu gate
                out, cache = affine_forward(layer_input,
                    self.params['W' + layer_index], self.params['b' + layer_index])
            else:
                # the intermediate hidden layers, with batch normalization or not
                if self.use_batchnorm:
                    out, cache = affine_bn_relu_forward(
                        layer_input,
                        self.params['W' + layer_index],
                        self.params['b' + layer_index],
                        self.params['gamma' + layer_index],
                        self.params['beta' + layer_index],
                        self.bn_params[i])
                else:
                    out, cache = affine_relu_forward(
                        layer_input,
                        self.params['W' + layer_index],
                        self.params['b' + layer_index])

                # add the dropout for the hidden layers, if there is any
                if self.use_dropout:
                    out, dropout_cache = dropout_forward(out, self.dropout_param)
                    # append the dropout cache
                    cache = (cache, dropout_cache)

            out_cache_dict['out' + layer_index] = out
            out_cache_dict['cache' + layer_index] = cache
            
            # the output of this layer becomes the input of next layer
            layer_input = out

        # Note: NO softmax on the final output
        scores = out

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
        
        # use the final output to calculate the loss
        loss, d_out = softmax_loss(out, y)

        # add L2 regularization
        if (self.reg):
            for i in range(self.num_layers):
                layer_index = str(i+1)
                penalty = np.sum(np.power(self.params['W' + layer_index], 2))
                loss += 0.5 * self.reg * penalty

        # iterate through the layers in reverse order
        for i in range(self.num_layers, 0, -1):
            layer_index = str(i)
            
            if i == self.num_layers :
                # the backward of the last layer does not need the ReLU gate
                dx, dw, db = affine_backward(d_out, out_cache_dict['cache' + layer_index])
            else:

                cache = out_cache_dict['cache' + layer_index]

                # check the dropout first, since it is right after ReLU.
                if self.use_dropout:
                    # decompose the cache
                    rest_cache, dropout_cache = cache
                    dx = dropout_backward(d_out, dropout_cache)

                    # pop out the dropout cache, and propapgate the derivative
                    cache = rest_cache
                    d_out = dx

                if self.use_batchnorm:
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(d_out, cache)

                    # save the additional parameters for batch normalization
                    grads['gamma' + layer_index] = dgamma
                    grads['beta' + layer_index] = dbeta
                else:
                    dx, dw, db = affine_relu_backward(d_out, cache)

            # the derivative output of the previous layer becomes the input of the next layer 
            d_out = dx
            
            # save the gradients for the affine layer
            grads['W' + layer_index] = dw
            grads['b' + layer_index] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
