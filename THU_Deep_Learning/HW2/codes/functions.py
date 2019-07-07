import numpy as np
from scipy import signal

# =============================================================================
# Main functions
# =============================================================================

def conv2d_forward(input, W, b, kernel_size, pad, stride=1):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    
#    print('Starting conv2d_forward... ')
    
    n, c_in, h_in, w_in = input.shape
    c_out = W.shape[0]
    # calculate output shape
    assert (h_in + 2 * pad - kernel_size) % stride == 0,'Invalid h_out'
    assert (w_in + 2 * pad - kernel_size) % stride == 0,'Invalid w_out'
    h_out = (h_in + 2 * pad - kernel_size)/stride + 1
    w_out = (w_in + 2 * pad - kernel_size)/stride + 1
    h_out = int(h_out)
    w_out = int(w_out)
        
    # pad input with 0s
    input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # compute output
    input_v = im2col(input_pad, h_out, w_out, c_in, kernel_size, stride=stride) # input_v.shape = (c_in * kernel_size^2, n * h_out * w_out)
    W_v = W.reshape(c_out, -1) # W_v.shape = (c_out, c_in * kernel_size^2)
    assert input_v.shape[0] == W_v.shape[-1],'input_v & W_v wrong dim'
    b_v = b.reshape(c_out, -1)
    
    output = np.matmul(W_v, input_v) + b_v # output.shape = (c_out, n * h_out * w_out)
    output = output.reshape(c_out, h_out, w_out, n)
    output = output.transpose(3, 0, 1, 2)
        
#    print('Ending conv2d_forward... ')
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad, stride=1, avgpool=False):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
#    print('Starting conv2d_backward... ')
    
    n, c_in, h_in, w_in = input.shape
    n_p, c_out, h_out, w_out = grad_output.shape
    assert n == n_p,'something wrong with n'
    
    grad_output_v = grad_output.transpose(1, 2, 3, 0)
    grad_output_v = grad_output_v.reshape(c_out, -1)
    
    # avgpool layer does not need grad_b & grad_W
    if avgpool:
        grad_b = None
        grad_W = None
    
    else:
        # compute grad_b
        grad_b = np.sum(grad_output, axis=(0, 2, 3))
    
        # compute grad_W
        input_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        input_v = im2col(input_pad, h_out, w_out, c_in, kernel_size)
        
        grad_W = np.matmul(grad_output_v, input_v.T)
        grad_W = grad_W.reshape(W.shape)

    # compute grad_input
    W_v = W.reshape(c_out, -1)
    grad_input_v = np.matmul(W_v.T, grad_output_v)
    grad_input = col2im(grad_input_v, input.shape, h_out, w_out, kernel_size, pad, stride=stride)
    assert grad_input.shape == input.shape,"grad_input shape wrong"

#    print('Ending conv2d_backward... ')
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    # stride is assumed to always be 1
    
    c_in = input.shape[1]
    W_ave = make_W_ave(kernel_size, c_in)
    b = np.zeros(c_in)

    # convolution forward
    output = conv2d_forward(input, W_ave, b, kernel_size, pad, stride=kernel_size)
    return output
  
   
def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    c_in = input.shape[1]
    W_ave = make_W_ave(kernel_size, c_in)
    b = np.zeros(c_in)

    # convolution backward
    grad_input, _, _ = conv2d_backward(input, grad_output, W_ave, b, kernel_size, pad, stride=kernel_size, avgpool=True)
    return grad_input


# =============================================================================
# Helper functions
# =============================================================================

def get_im2col_indices(h_out, w_out, c_in, kernel_size, stride=1):
    # Get the indices of the input matrix to build the im2col vector
    ii = np.tile(np.repeat(np.arange(kernel_size), kernel_size), c_in)
    iii = stride * np.repeat(np.arange(h_out), w_out)
    jj = np.tile(np.arange(kernel_size), kernel_size * c_in)
    jjj = stride * np.tile(np.arange(w_out), h_out)
    i = ii.reshape(-1, 1) + iii.reshape(1, -1)
    j = jj.reshape(-1, 1) + jjj.reshape(1, -1)
    c = np.repeat(np.arange(c_in), kernel_size * kernel_size).reshape(-1, 1)
    
    return c, i, j


def im2col(input, h_out, w_out, c_in, kernel_size, stride=1):
    # input is a 4d matrix, all other arguments are scalars
    # Stride is implicitly assumed to be 1
    
    # Get the input for the relevant indices
    c, i, j = get_im2col_indices(h_out, w_out, c_in, kernel_size, stride)
    input_v = input[:, c, i, j]
    
    # Shape into desired vector
    input_v = input_v.transpose(1, 2, 0)
    input_v = input_v.reshape(kernel_size * kernel_size * c_in, -1)
    return input_v


def col2im(input_v, input_shape, h_out, w_out, kernel_size, pad=0, stride=1):
    # input_v is a 2d matrix, input_shape is a tuple of length 4, the rest are scalars
    
    n, c_in, h_in, w_in = input_shape
    input_pad_shape = (n, c_in, h_in + 2 * pad, w_in + 2 * pad)
    input_pad = np.zeros(input_pad_shape, dtype = input_v.dtype)
    
    # Get the input for the relevant indices
    c, i, j = get_im2col_indices(h_out, w_out, c_in, kernel_size, stride)
  
    input_r = input_v.reshape(c_in * kernel_size * kernel_size, -1, n)
    input_r = input_r.transpose(2, 0, 1)
    np.add.at(input_pad, (slice(None), c, i, j), input_r)
    
    # Discard padding if no padding
    if pad == 0:
        input_f = input_pad
    else:
        input_f = input_pad[:, :, pad:-pad, pad:-pad]
    return input_f


def make_W_ave(kernel_size, c_in):
    # make a custom W matrix to convolve with input, used in avgpool layer
    pooling_size = kernel_size * kernel_size
    W_ave = []
    for i in range(c_in):
        before = np.zeros(pooling_size * i)
        ave = np.repeat(1/pooling_size, pooling_size)
        after = np.zeros(pooling_size * (c_in - i - 1))
        W_ave = np.concatenate((W_ave, before, ave, after))
    W_ave = W_ave.reshape(c_in, c_in, kernel_size, kernel_size)
    return W_ave