#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Naifu 2018280351

Script to train word embedding with Wikipedia corpus and test with WordSim353, built with gensim
Adapted from tensorflow word2vec tutorial https://rare-technologies.com/word2vec-tutorial/

"""
import re
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import FastText

from nltk.corpus import stopwords

# Read the data into a list of strings.
def read_data(filename):
    """Extract the file as a list of words."""
    with open(filename) as f:
        input_text = tf.compat.as_str(f.read())
        
    # remove parenthesis 
    input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
    # store as list of sentences
    sentences_strings_ted = []
    for line in input_text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    # store as list of lists of words
    sentences_ted = []
    for sent_str in sentences_strings_ted:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        sentences_ted.append(tokens)
    
#    # Convert to lower case, remove punctuations, non-alphanumeric and stop words
#    tokens = [w.lower() for w in data]
#    table = str.maketrans('', '', string.punctuation)
#    stripped = [w.translate(table) for w in tokens]
#    words = [word for word in stripped if word.isalpha()]
#    stop_words = set(stopwords.words('english'))
#    data = [w for w in words if not w in stop_words]    
#    return data

#model_ted = FastText(sentences_ted, size=100, window=5, min_count=5, workers=4,sg=1)

read_data('wiki/tryaa')
