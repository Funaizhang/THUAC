# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:38:06 2018

@author: zhangnaifu
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def ReLU(x):
    a = []
    for item in x:
        a.append(max(0,item))
    return a

def linear(x):
    a = []
    for item in x:
        a.append(item)
    return a

def tanh(x):
    a = []
    for item in x:
        a.append((math.exp(item)-math.exp(-item))/(math.exp(item)+math.exp(-item)))
    return a


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5., 5., 0.1)
sig = sigmoid(x)
tan = tanh(x)
lin = linear(x)
re = ReLU(x)

# plot with various axes scales
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x, sig)
plt.yscale('linear')
plt.title('Sigmoid')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, tan)
plt.yscale('linear')
plt.title('tanh')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, re)
plt.yscale('linear')
plt.title('ReLU')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, lin)
plt.yscale('linear')
plt.title('Linear')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
#                    wspace=0.35)