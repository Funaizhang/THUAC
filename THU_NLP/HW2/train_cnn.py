#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:34:52 2019

@author: Naifu 2018280351
"""

import os
import numpy as np
import csv
from keras.layers import Dense, Conv1D, MaxPooling1D, Embedding, Dropout, GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import load_model
from utils import *
#from test import eval_models

BASE_DIR = os.getcwd()
GLOVE_PATH = os.path.join(BASE_DIR, 'glove.6B.300d.txt')
TRAIN_PATH = os.path.join(BASE_DIR, 'data/train.txt')
VAL_PATH = os.path.join(BASE_DIR, 'data/dev.txt')
TEST_PATH = os.path.join(BASE_DIR, 'data/test.txt')

# Training configuration
config = {
    'MAX_NO_SENTENCE': None,
    'MAX_SEQUENCE_LENGTH': 60, # max 52 words per sentence, rounded up for cleaner maxpooling later
    'EMBEDDING_DIM': 300,
    'LABELS': 5,
    'batch_size': 128,
    'max_epoch': 10,
}

trainables = [False]
hidden_layers = [3]
hidden_sizes = [512]
dropout_rs = [0.2]
regularizes = [0.05]


## load GLOVE embeddings, returns a dict embeddings_index: (word, embedding)
#embeddings_index = read_embeddings(GLOVE_PATH)

# get train, val, test data
paths = [TRAIN_PATH, VAL_PATH, TEST_PATH]
word_index, train_x, train_y, val_x, val_y, test_x, test_y = read_data(paths, max_len=config['MAX_SEQUENCE_LENGTH'])

vocab_size = len(word_index) + 1
embedding_matrix = make_w_matrix(embeddings_index, word_index, embedding_dim=config['EMBEDDING_DIM']) # embedding_matrix: shape = vocab_size in train_x * EMBEDDING_DIM

    
# define and build CNN model
def build_cnn(config, trainable=False, hidden_layer=1, dropout_r=0.2, hidden_size=256, regularize=0.0):

    model_name = 'trainable_{}-hiddenlayer_{}-dropout_{}-hiddensize_{}-regularize{}'.format(trainable, hidden_layer, dropout_r, hidden_size, regularize)

    # define and build CNN model
    model = Sequential(name=model_name)
    model.add(Embedding(vocab_size, config['EMBEDDING_DIM'], weights=[embedding_matrix], input_length=config['MAX_SEQUENCE_LENGTH'], trainable=trainable, name="embedding"))
    for i in range(hidden_layer):
        model.add(Conv1D(hidden_size, 5, activation='relu', padding='same', name="conv1D_{}".format(i)))
        model.add(MaxPooling1D(3, name="maxpool1D_{}".format(i)))
    model.add(GlobalMaxPooling1D(name="maxpool1D_{}".format(hidden_layer)))
    if dropout_r != None:
        model.add(Dropout(dropout_r, name="dropout"))
    model.add(Dense(config['LABELS'], activation='softmax', kernel_regularizer=l2(regularize), bias_regularizer=l2(regularize)))
    
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    # summarize the model
    print(model.summary())
    
    return model


def train_model(config, model):
    
    # Configure checkpoints to save weights after each epoch if val_acc improves
    tensorboard = TensorBoard(log_dir='weights/logs/{}'.format(model.name), write_graph=False)
    weights_path = "weights/{}".format(model.name)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    filepath = os.path.join(weights_path, "epoch{epoch:02d}-val_acc{val_acc:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    csv_logger = CSVLogger('logs/{}.log'.format(model.name))
    
    callbacks_list = [tensorboard, checkpoint, csv_logger]
    
    # fit the model
    history = model.fit(
            train_x, 
            train_y, 
            validation_data = (val_x, val_y), 
            epochs = config['max_epoch'], 
            batch_size = config['batch_size'],
            callbacks=callbacks_list,
            verbose=1)
    
    return history, weights_path


def eval_models(folder_path, test_x, test_y):
    # evaluates all the weights saved for a model in a model folder, returns the best test acc
    loss_list = []
    acc_list = []
    model_paths = os.listdir(folder_path)
    
    for model_path in model_paths:
        if model_path[0] == '.':
            continue
        else:
            model_path = os.path.join(folder_path, model_path)
            model = load_model(model_path)
            
            # evaluate the model
            loss, acc = model.evaluate(test_x, test_y, verbose=1)
            loss_list.append(loss)
            acc_list.append(acc)
       
    test_loss = min(loss_list)
    test_acc = max(acc_list)
    
    print('{0}: acc {1}%'.format(model.name, round(test_acc*100,2)))
    
    return round(test_acc,4), test_loss


# train all model variations
for trainable in trainables:
    for hidden_layer in hidden_layers:
        for dropout_r in dropout_rs:
            for hidden_size in hidden_sizes:
                for regularize in regularizes:
                    
                    if not os.path.exists('weights'):
                        os.makedirs('weights')
                    if not os.path.exists('logs'):
                        os.makedirs('logs')
                    if not os.path.exists('plots'):
                        os.makedirs('plots')
                
                    # build and train model
                    model = build_cnn(config, trainable, hidden_layer, dropout_r, hidden_size, regularize)
                    history, weights_path = train_model(config, model)
                    train_acc, val_acc = history.history['acc'], history.history['val_acc']
                    train_loss, val_loss = history.history['loss'], history.history['val_loss']
                    plot_loss_acc(config['max_epoch'], train_acc, val_acc, train_loss, val_loss, model.name)
                    
                    # evaluate model on test set
                    test_acc_best, test_loss_best = eval_models(weights_path, test_x, test_y)
                    
                    # save all metrics in csv
                    train_acc_best = max(train_acc)
                    val_acc_best = max(val_acc)
                    train_loss_best = min(train_loss)
                    val_loss_best = min(val_loss)
                    
                    # write to csv
                    row = [model.name, train_acc_best, train_loss_best, val_acc_best, val_loss_best, test_acc_best, test_loss_best]
                    with open('results.csv', 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(row)
                    csvFile.close()
