#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:21:12 2019

@author: Naifu 2018280351
"""

import numpy as np
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# read embeddings
def read_embeddings(path):
    # first, build index mapping words in the embeddings set to their embedding vector
    print('Indexing word vectors.')
    embeddings_index = dict()
    
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


# find the max no. of words per line; used for padding later
def words_per_line(path):
    words_no_list = []
    with open(path) as f:
        for line in f:
            sentence = re.sub("\d+", "", line)
            words_no = len(re.findall(r'\w+', sentence))
            words_no_list.append(words_no)
    return max(words_no_list)

            
# read and format train, val & test data
def read_data(paths, max_len=60, max_lines=None):
    print('Reading data.')
    assert len(paths) == 3,'1st argument should be train, val, test data paths'
    lines = 0
    sentences = []
    all_sentences = []
    all_labels = []
    
    for path in paths:
        with open(path) as f:
            file_sentences = []
            file_labels = []
            for line in f:
                # max_lines = no. of lines to read
                if max_lines != None and lines >= max_lines:
                    break
                else:
                    lines += 1
                    
                    # class label for the sentence is always the 2nd element of the line
                    label = line[1]
                    file_labels.append(int(label))
                    
                    # parse the whole line for sentence
                    sentence = re.sub("\d+", "", line)
                    sentence = re.findall(r'\w+', sentence)
                    sentence = " ".join(sentence)
                    sentences.append(sentence)
                    file_sentences.append(sentence)
    
            all_sentences.append(file_sentences)
            all_labels.append(file_labels)
    
    train_x = all_sentences[0]
    val_x = all_sentences[1]
    test_x = all_sentences[2]
    train_y = all_labels[0]
    val_y = all_labels[1]
    test_y = all_labels[2]
    
    # convert sentences to padded sequences
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(sentences)
    word_index = t.word_index
    
    # integer encode the sentences; pad sentences to a max length of max_len words
    train_encoded = t.texts_to_sequences(train_x)
    train_padded = pad_sequences(train_encoded, maxlen=max_len, padding='post')
    val_encoded = t.texts_to_sequences(val_x)
    val_padded = pad_sequences(val_encoded, maxlen=max_len, padding='post')
    test_encoded = t.texts_to_sequences(test_x)
    test_padded = pad_sequences(test_encoded, maxlen=max_len, padding='post')
    
#    # format the labels
#    labels = to_categorical(np.asarray(labels))
    
    print('Read %s sentences.' % lines)
    return word_index, train_padded, train_y, val_padded, val_y, test_padded, test_y


def make_w_matrix(embeddings_index, word_index, embedding_dim=300):
    vocab_size = len(word_index) + 1
    
    # create a weight matrix for words in padded_sequences
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    
    return embedding_matrix           


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