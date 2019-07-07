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
    
    
def make_plot(epochs, train, test, model_name, loss_name, loss=True):
# =============================================================================
#    make_plot plots the training loss/accuracy & test loss/accuracy wrt each epoch
#    so for each (model, loss) variation, we call make_plot twice to plot loss & accuracy respectively
    
#    epochs is a single number
#    train, test are arrays of length = epochs
# =============================================================================
    
    loss_or_accuracy = 'loss'
    log_or_linear = 'log'
    if not loss:
        loss_or_accuracy = 'accuracy'
        log_or_linear = 'linear'
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["#2A6EA6", "#FFA933"]
              
    # Plot for training loss/accuracy
    ax.plot(np.arange(epochs), 
#            [0.1, 0.1, 0.5, 0.05],
            train,
            color=colors[0],
            label="training %s" % loss_or_accuracy)
    
    # Plot for test loss/accuracy
    ax.plot(np.arange(epochs), 
#            [0.1, 0.2, 0.05, 0.05],
            test,
            color=colors[1],
            label="test %s" % loss_or_accuracy)
    
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel('Epochs no.')
    ax.set_title('{0} from {1} using {2}'.format(loss_or_accuracy, model_name, loss_name))
    ax.set_yscale(log_or_linear)
    plt.legend(loc="upper right")
    fig_filename = '{0}_{1}_{2}.png'.format(model_name, loss_name, loss_or_accuracy)
    plt.savefig(os.path.join("plots", fig_filename))
    
    
    

