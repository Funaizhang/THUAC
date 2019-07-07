from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter 
import numpy as np
import os

def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print('[' + display_now + ']' + ' ' + msg)


def plot_loss_acc(epochs, train_acc, test_acc, train_loss, test_loss, model_name):
# =============================================================================
#    loss_acc_plot plots the training loss/accuracy & test loss/accuracy wrt each epoch
#    so for each (model, loss) variation, we call make_plot twice to plot loss & accuracy respectively
    
#    epochs is a scalar number
#    train, test are arrays of length = epochs
# =============================================================================
    
    log_or_linear = 'linear'
    location = "upper right"
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["#2A6EA6", "#FFA933"]
    
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel('Epochs no.')
    ax.set_title('{}'.format(model_name))
    ax.set_yscale(log_or_linear)
    ax.set_ylabel('acc', color="#000000")
    
    # Plot for training acc
    ln1 = ax.plot(np.arange(epochs), 
            train_acc,
            color=colors[0],
            label="train acc")
    
    # Plot for test acc
    ln2 = ax.plot(np.arange(epochs), 
            test_acc,
            color=colors[1],
            label="val acc")
    
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    
    ax2 = ax.twinx() # instantiate a second y-axes that shares the same x-axis
    ax2.set_yscale(log_or_linear)
    ax2.set_ylabel('loss', color="#000000")
                  
    # Plot for training loss
    ln3 = ax2.plot(np.arange(epochs), 
            train_loss,
            color=colors[0],
            dashes=[5, 2],
            label="train loss")
    
    # Plot for test loss
    ln4 = ax2.plot(np.arange(epochs), 
            test_loss,
            color=colors[1],
            dashes=[5, 2],
            label="val loss")
    
    lns = ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=location)

    # save fig
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    fig_filename = '{}.png'.format(model_name)
    fig.tight_layout()
    plt.savefig(os.path.join("plots", fig_filename))
    
    plt.clf()
    plt.cla()
    plt.close()