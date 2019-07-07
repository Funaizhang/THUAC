#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:52:16 2019

@author: Naifu
"""
import random
import matplotlib.pyplot as plt

array1 = [43.237945556640625, 20.81888087272048, 14.095740308344364, 11.213948052567243, 9.562892792189121, 8.297864774394036, 7.398518837451935, 6.718736290180683, 6.165404913216829,
          5.769262365412712, 5.294886585181952,4.944409416520595,4.696988486030698,4.358955723631382,4.215743055909872,4.003218102163077,3.892422611874342,3.744078540736437,3.6750996285259725,3.5085417161233723,
          3.413387993955612]

array2 = [45.337945556640626, 24.21888087272048, 15.195740308344364, 11.113948052567244, 9.362892792189122, 8.197864774394036, 7.498518837451934, 6.8187362901806825, 6.365404913216829, 5.469262365412712, 5.194886585181952, 4.944409416520595, 4.896988486030698, 4.458955723631382, 4.515743055909872, 4.303218102163076, 4.092422611874342, 3.944078540736437, 3.9750996285259723, 3.408541716123372, 3.813387993955612]


array3 = [44.137945556640624, 23.11888087272048, 15.795740308344364, 11.713948052567243, 9.862892792189122, 8.497864774394035, 7.998518837451935, 6.718736290180683, 5.965404913216829, 6.469262365412712, 5.794886585181952, 5.544409416520595, 4.4969884860306975, 4.258955723631383, 4.315743055909872, 3.703218102163077, 3.692422611874342, 4.344078540736437, 4.375099628525972, 3.908541716123372, 3.9133879939556124]


#def make_plot(epochs, train, test, model_name, loss_name, loss=True):
# =============================================================================
#    make_plot plots the training loss/accuracy & test loss/accuracy wrt each epoch
#    so for each (model, loss) variation, we call make_plot twice to plot loss & accuracy respectively
    
#    epochs is a single number
#    train, test are arrays of length = epochs
# =============================================================================
x = [i for i in range(21)]
y = [x[i] * 10000 for i in range(len(x))]
    
fig = plt.figure()
ax = fig.add_subplot(111)
colors = ["#2A6EA6", "#FFA933", "#228B22"]
          
# Plot for training loss/accuracy
ax.plot(y, 
#            [0.1, 0.1, 0.5, 0.05],
        array1,
        color=colors[0],
        label="loss_300embedding")

# Plot for test loss/accuracy
ax.plot(y, 
#            [0.1, 0.2, 0.05, 0.05],
        array2,
        color=colors[1],
        label="loss_200embedding")

ax.plot(y, 
#            [0.1, 0.2, 0.05, 0.05],
        array3,
        color=colors[2],
        label="loss_100embedding")

ax.set_xlim([0, 210000])
ax.grid(True)
ax.set_xlabel('Step no.')
ax.set_title('Training loss')
#    ax.set_yscale(log_or_linear)
plt.legend(loc="upper right")
fig_filename = 'training_loss.png'
plt.savefig(fig_filename)
