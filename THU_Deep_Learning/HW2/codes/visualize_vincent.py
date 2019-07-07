# =============================================================================
# This is Vincent's code to visualize. 
# Right now this approach does not work, network needs to save weights
# I also don't like the way it ad hoc generates output
# =============================================================================

import numpy as np
from load_data import load_mnist_4d
from utils import vis_square

_, test_data, _, test_label = load_mnist_4d('data')

data = np.array([test_data[test_label == x][0] for x in np.arange(10)])

W = np.load('./weights-save/conv1-W-99.npy')
b = np.load('./weights-save/conv1-b-99.npy')

output = conv2d_forward(data, W, b, W.shape[2], W.shape[2] // 2).transpose((1, 0, 2, 3)).clip(0)
output = output.reshape(output.shape[0] * output.shape[1], output.shape[2], output.shape[3])

vis_square(W.squeeze(), n=1, figsize=5)

vis_square(output, n=10, figsize=10)