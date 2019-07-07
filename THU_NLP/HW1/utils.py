#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:52:51 2019

@author: Naifu
"""

import numpy as np
import string
import collections
from nltk.corpus import stopwords
import tensorflow as tf
import csv
import pandas as pd

"""
Helper functions to load and process data

"""
# Step 1: Read the data into a list of strings, also clean the data
def read_data(filename, num_char=800000000):
    """Extract the file as a list of words. Only read the first num_char characters."""
    print('Reading data.')
    with open(filename) as f:
        data = tf.compat.as_str(f.read(num_char)).split()
    
    # Convert to lower case, remove punctuations, non-alphanumeric and stop words
    tokens = [w.lower() for w in data]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    vocabulary = [w for w in words if not w in stop_words]
    
    print('Data size', len(vocabulary))
    return vocabulary



def read_sim_list(filename, valid_dict=None):
    df = pd.read_csv(filename)
    test_list = []
    for word1, word2, score in df.values:
        id1 = valid_dict.get(word1)
        id2 = valid_dict.get(word2)
        if id1 is not None and id2 is not None:
            test_list.append([id1, id2, score])
    test_list = np.array(test_list)
    print('test_list is: ' + str(test_list.size))
    words_list, score_list = test_list[:, [0, 1]], test_list[:, 2]
    return words_list, score_list

    

# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary



#vocabulary = read_data('./wiki_corpus/xaa')
#print('Data size', len(vocabulary))
## Filling 4 global variables:
## data - list of codes (integers from 0 to vocabulary_size-1). This is the original text but words are replaced by their codes
## count - map of words(strings) to count of occurrences
## dictionary - map of words(strings) to their codes(integers)
## reverse_dictionary - maps codes(integers) to words(strings)
#data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, 50000)
#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#del vocabulary
#
## Convert simwords into a list of tuples of format [(word1, word2)] and [score]
## test_words Only contains valid words
#print(read_sim_list('./wordsim353/combined.csv', dictionary))
    