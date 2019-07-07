from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np


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