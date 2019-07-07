#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:31:18 2019

@author: Naifu 2018280351
"""

from keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, Dropout, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.regularizers import l2

# define and build CNN model
def build_cnn(config, trainable=False, hidden_layer=1, dropout_r=0.2, hidden_size=256, regularize=0.0):

    model_name = 'trainable_{}-hiddenlayer_{}-dropout_{}-hiddensize_{}-regularize{}'.format(trainable, hidden_layer, dropout_r, hidden_size, regularize)

    # define and build CNN model
    model = Sequential(name=model_name)
    model.add(Embedding(
                    vocab_size, 
                    config['EMBEDDING_DIM'], 
                    weights=[embedding_matrix], 
                    input_length=config['MAX_SEQUENCE_LENGTH'], 
                    trainable=trainable, 
                    name="embedding"))
    
    for i in range(hidden_layer):
        model.add(Conv1D(
                hidden_size, 
                5, 
                activation='relu', 
                padding='same', 
                name="conv1D_{}".format(i)))
        model.add(MaxPooling1D(3, name="maxpool1D_{}".format(i)))
        
    model.add(GlobalMaxPooling1D(name="maxpool1D_{}".format(hidden_layer)))
    
    if dropout_r != None:
        model.add(Dropout(dropout_r, name="dropout"))
    
    model.add(Dense(config['LABELS'], 
                    activation='softmax', 
                    kernel_regularizer=l2(regularize), 
                    bias_regularizer=l2(regularize)))
    
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # summarize the model
    print(model.summary())
    
    return model