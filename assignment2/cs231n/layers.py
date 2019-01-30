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
    flatted_x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
    out = flatted_x @ w + b
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
      - b: Biases, of shape (M,)

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
    flatted_x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
    dw = flatted_x.T @ dout
    dx = dout @ w.T
    dx = dx.reshape(*x.shape)
    db = np.sum(dout, axis = 0)
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
    out = np.maximum(0, x)
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
    dx = (x > 0) * dout
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
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        means = np.mean(x, axis=0)
        variances = np.var(x, axis=0)
        variances += eps
        normalized_x = (x - means)/np.sqrt(variances)

        out = normalized_x * gamma
        out += beta

        running_mean = momentum*running_mean + (1-momentum)*means
        running_var = momentum*running_var + (1-momentum)*variances

        cache = (x, means, N, normalized_x, variances, gamma)
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
        out = (x - running_mean)/np.sqrt(running_var)
        out *= gamma
        out += beta
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
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    (x, means, N, normalized_x, variances, gamma) = cache
    dl_dbeta = np.sum(dout, axis=0)
    dl_dgamma = np.sum(normalized_x * dout, axis=0)
    dl_dxhat = gamma * dout 
    
    #partial derivatives of normalization function
    dl_dmu_2 = np.sum(-(1/np.sqrt(variances)) * dl_dxhat, axis=0)
    dl_ds = np.sum(- 0.5 * (x - means) * np.power(variances, -3/2) * dl_dxhat, axis=0)
    dl_dx_1 = dl_dxhat / np.sqrt(variances)

    #partial derivatives of variance function
    dl_dmu_1 = np.sum(-2*x + 2*means, axis=0) * dl_ds / N
    dl_dx_2 = (2*x - 2*means) * dl_ds / N

    #derivative of mean function (summing partial derivatives for mean)/
    dl_dmu = dl_dmu_1 + dl_dmu_2
    dl_dx_3 = dl_dmu / N

    #summing partial derivatives for x
    dl_dx = dl_dx_1 + dl_dx_2 + dl_dx_3

    dbeta = dl_dbeta
    dgamma = dl_dgamma
    dx = dl_dx

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
    See the jupyter notebook for more hints.
     
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
    (_, _, N, normalized_x, variances, gamma) = cache
    dgamma = np.sum(normalized_x * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    #see https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
    dx = (gamma/N) / np.sqrt(variances) * (N * dout - dgamma * normalized_x - dbeta) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    N, _ = x.shape
    
    x_T = x.T
    means = np.mean(x_T, axis=0)
    variances = np.var(x_T, axis=0)
    variances += eps
    normalized_x = (x_T - means)/np.sqrt(variances)
    normalized_x = normalized_x.T
    out = normalized_x * gamma
    out += beta
    cache = (x, means, normalized_x, variances, gamma)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    _, D = dout.shape
    (x, means, normalized_x, variances, gamma) = cache

    variances = variances[:,np.newaxis]
    means = means[:,np.newaxis]
    gamma = gamma[np.newaxis,:]

    dl_dbeta = np.sum(dout, axis=0) #[N,]
    dl_dgamma = np.sum(normalized_x * dout, axis=0) #[N,]
    dl_dxhat = gamma * dout #[N,D]

    #partial derivatives of normalization function
    dl_dmu_2 = np.sum(-(1/np.sqrt(variances)) * dl_dxhat, axis=1) #[N,]
    dl_ds = np.sum(- 0.5 * (x - means) * np.power(variances, -3/2) * dl_dxhat, axis=1) #[N,]
    dl_dx_1 = dl_dxhat / np.sqrt(variances) #[N,D]
    

    #partial derivatives of variance function
    dl_dmu_1 = np.sum(-2*x + 2*means, axis=1) * dl_ds / D #[N,]
    dl_dx_2 = (2*x - 2*means) * dl_ds[:,np.newaxis] / D #[N,D]
    

    #derivative of mean function (summing partial derivatives for mean)/
    dl_dmu = dl_dmu_1 + dl_dmu_2
    dl_dx_3 = dl_dmu / D

    #summing partial derivatives for x
    dl_dx = dl_dx_1 + dl_dx_2 + dl_dx_3[:,np.newaxis]

    dbeta = dl_dbeta
    dgamma = dl_dgamma
    dx = dl_dx
    
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
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
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
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
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
        dx = mask*dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

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
    
    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    H_ = int(1 + (H + 2*pad - HH)/stride)
    W_ = int(1 + (W + 2*pad - WW)/stride)
    out = np.zeros((N,F,H_,W_))

    boundaries_x = [(x,x+WW) for x in range(0, W, stride) ]
    boundaries_y = [(x,x+HH) for x in range(0, H, stride) ]
    for i in range(0,N):
        sample = x[i,:,:,:]
        padded_sample = np.pad(sample, pad, "constant")[pad:-pad]
        for u, (x1,x2) in enumerate(boundaries_x):
            for v, (y1,y2) in enumerate(boundaries_y):
                receptive_field = padded_sample[:, y1:y2, x1:x2]
                for f in range(0,F):
                    conv = np.sum(receptive_field*w[f,:,:,:]) + b[f]
                    out[i,f,v,u] = conv    
                    
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
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, Hh, Hw = dout.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    padded_x = np.pad(x, pad, "constant")[pad:-pad,pad:-pad]

    dw = np.zeros_like(w)
    for f in range(0, F):
        for c in range(0, C):
            for i in range(0,HH):
                for j in range(0, WW):
                    sub_xpad = padded_x[:, c, i:i + Hh * stride:stride, j:j + Hw * stride:stride]
                    dw[f, c, i, j] = np.sum(dout[:, f, :, :] * sub_xpad)

    db = np.zeros_like(b)
    for f in range(0, F):
        db[f] = np.sum(dout[:, f, :, :])

    boundaries_x_w = [(x,x+HH) for x in range(0, H, stride) ]
    boundaries_y_w = [(x,x+WW) for x in range(0, W, stride) ]
    dx = np.zeros_like(padded_x)
    for n in range(0, N):
        for f in range(F):
            for i, (x1,x2) in enumerate(boundaries_x_w):
                for j, (y1,y2) in enumerate(boundaries_y_w):
                    dx[n, :, x1:x2, y1:y2] += w[f] * dout[n, f, i, j]

    dx = dx[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    Hh, Hw = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape
    H_ = int(1 + (H-Hh) / stride)
    W_ = int(1 + (W-Hw) / stride)
    out = np.zeros((N,C,H_,W_))

    boundaries_x = [(x,x+Hh) for x in range(0, H, stride) ]
    boundaries_y = [(x,x+Hw) for x in range(0, W, stride) ]
    for n in range(0,N):
        for c in range(0,C):
            for i, (x1,x2) in enumerate(boundaries_x):
                for j, (y1,y2) in enumerate(boundaries_y):
                    out[n,c,i,j] = np.max(x[n,c,x1:x2,y1:y2])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    Hh, Hw = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    N, C, H, W = x.shape
    H_ = int(1 + (H-Hh) / stride)
    W_ = int(1 + (W-Hw) / stride)
    dx = np.zeros_like(x)

    boundaries_x = [(x,x+Hh) for x in range(0, H, stride) ]
    boundaries_y = [(x,x+Hw) for x in range(0, W, stride) ]

    for n in range(0,N):
        for c in range(0,C):
            for i, (x1,x2) in enumerate(boundaries_x):
                for j, (y1,y2) in enumerate(boundaries_y):
                    max_element = np.max(x[n,c,x1:x2,y1:y2])
                    dx[n,c,x1:x2,y1:y2] = (x[n,c,x1:x2,y1:y2] == max_element) * dout[n,c,i,j]
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
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    (N, C, H, W) = x.shape
    out = np.zeros_like(x)
    bn_out, cache = batchnorm_forward(x.transpose(1,0,2,3).reshape((C, N*H*W)).T, gamma, beta, bn_param)
    out = bn_out.T.reshape(C,N,H,W).transpose(1,0,2,3)
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
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    (N, C, H, W) = dout.shape
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(1,0,2,3).reshape((C, N*H*W)).T, cache)
    dx = dx.T.reshape(C,N,H,W).transpose(1,0,2,3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N,C,H,W = x.shape

    x_T = x.reshape((N,C*H*W)).T
    x_T = np.split(x_T,G)

    normalized_x = np.empty((0,N))
    means = np.empty((0,N))
    variances = np.empty((0,N))
    for part_x in x_T:
        part_means = np.mean(part_x, axis=0)

        part_variances = np.var(part_x, axis=0)
        part_variances += eps
    
        normalized_part_x = (part_x - part_means)/np.sqrt(part_variances)
        normalized_x = np.vstack((normalized_x, normalized_part_x))
        means = np.vstack((means, part_means))
        variances = np.vstack((variances, part_variances))

    normalized_x = normalized_x.T.reshape((N,C,H,W))
    out = normalized_x * gamma
    out += beta
    cache = (x, means, normalized_x, variances, gamma, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

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
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N,C,H,W = dout.shape
    (x, means, normalized_x, variances, gamma, G) = cache

    dl_dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) #[1,C,1,1]
    dl_dgamma = np.sum(normalized_x * dout, axis=(0,2,3), keepdims=True) #[1,C,1,1]
    
    dl_dxhat = gamma * dout
    dl_dxhat = dl_dxhat.reshape((N,C*H*W))
    x = x.reshape((N,C*H*W))

    dl_dx = np.empty((N,0))
    for i, (part_dl_dxhat, part_x) in enumerate(zip(np.split(dl_dxhat, G, axis=1), np.split(x, G, axis=1))):
        part_variances = variances[i][:,np.newaxis]
        part_means = means[i][:,np.newaxis]
        part_dl_dmu_2 = np.sum(-(1/np.sqrt(part_variances)) * part_dl_dxhat, axis=1)
        part_dl_ds = np.sum(- 0.5 * (part_x - part_means) * np.power(part_variances, -3/2) * part_dl_dxhat, axis=1)
        part_dl_dx_1 = part_dl_dxhat / np.sqrt(part_variances) 

        _, D = part_dl_dxhat.shape
        part_dl_dmu_1 = np.sum(-2*part_x + 2*part_means, axis=1) * part_dl_ds / D
        part_dl_dx_2 = (2*part_x - 2*part_means) * part_dl_ds[:,np.newaxis] / D

        part_dl_dmu = part_dl_dmu_1 + part_dl_dmu_2
        part_dl_dx_3 = part_dl_dmu / D

        part_dl_dx = part_dl_dx_1 + part_dl_dx_2 + part_dl_dx_3[:,np.newaxis]
        dl_dx = np.hstack((dl_dx, part_dl_dx))

    dl_dx = dl_dx.reshape((N,C,H,W))
    
    dbeta = dl_dbeta
    dgamma = dl_dgamma
    dx = dl_dx    
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
