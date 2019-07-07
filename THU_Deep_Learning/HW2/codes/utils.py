from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os


def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)
    
    
def plot_loss_acc(epochs, train, test, model_name, loss_name, loss=True):
# =============================================================================
#    loss_acc_plot plots the training loss/accuracy & test loss/accuracy wrt each epoch
#    so for each (model, loss) variation, we call make_plot twice to plot loss & accuracy respectively
    
#    epochs is a scalar number
#    train, test are arrays of length = epochs
# =============================================================================
    
    loss_or_accuracy = 'loss'
    log_or_linear = 'log'
    location = "upper right"
    if not loss:
        loss_or_accuracy = 'accuracy'
        log_or_linear = 'linear'
        location = "lower right"
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["#2A6EA6", "#FFA933"]
              
    # Plot for training loss/accuracy
    ax.plot(np.arange(epochs), 
            train,
            color=colors[0],
            label="training %s" % loss_or_accuracy)
    
    # Plot for test loss/accuracy
    ax.plot(np.arange(epochs), 
            test,
            color=colors[1],
            label="test %s" % loss_or_accuracy)
    
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel('Epochs no.')
    ax.set_title('{0} from {1} using {2}'.format(loss_or_accuracy, model_name, loss_name))
    ax.set_yscale(log_or_linear)
    plt.legend(loc=location)

    # save fig
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    fig_filename = '{0}_{1}_{2}.png'.format(model_name, loss_name, loss_or_accuracy)
    plt.savefig(os.path.join("plots", fig_filename))
    
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_vis(data, model_name, loss_name, layer_name):
    """Takes in data of shape (n, height, width) or (n, height, width, 3) and visualize each in a grid of sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data, cmap='Greys_r'); plt.axis('off')
    # save fig
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    fig_filename = 'visuals_{0}_{1}_{2}.png'.format(model_name, loss_name, layer_name)
    plt.savefig(os.path.join("plots", fig_filename))


def wb_save(data, model_name, loss_name, epoch, layer_name, W_or_b='W'):
    # save weights and biases
    if not os.path.exists("weights"):
        os.makedirs("weights")
    weights_filename = '{0}_{1}_{2}_{3}_{4}.png'.format(model_name, loss_name, epoch, layer_name, W_or_b)
    np.save(os.path.join("weights", weights_filename), data)