from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO, plot_loss_acc, plot_vis
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
import time

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network('CNN_test')
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.1))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 0.1))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropy')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 2,
    'disp_freq': 100,
    'layer_vis': 'relu1'
}


train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
plot_visual = []

start_time = time.time()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    # save loss & accuracy data for training epoch
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    test_loss, test_acc, visual = test_net(model, loss, test_data, test_label, config['batch_size'], epoch, config['layer_vis'])
    # save loss & accuracy data for test epoch
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # plot visual during testing of the last epoch
    if epoch == (config['max_epoch']-1):
        plot_visual = visual[0:25, :, :]

end_time = time.time()

plot_loss_acc(config['max_epoch'], train_loss_list, test_loss_list, model.name, loss.name, loss=True)
plot_loss_acc(config['max_epoch'], train_acc_list, test_acc_list, model.name, loss.name, loss=False)
plot_vis(plot_visual, model.name, loss.name, config['layer_vis'])
print('Time per epoch = ' + str((end_time-start_time)/config['max_epoch']) + 's')
