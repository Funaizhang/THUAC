# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
@author: Naifu 2018280351

Script to train word embedding with Wikipedia corpus and test with WordSim353, built on Tensorflow
Adapted from tensorflow word2vec tutorial https://www.tensorflow.org/tutorials/representation/word2vec
As the original word2vec.py file was already heavily commented, I have added comments sparsely to the existing code. 
New lines of code are commented.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import csv
from tempfile import gettempdir
import string
from nltk.corpus import stopwords


import numpy as np 
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

data_index = 0

vocabulary_size = 50000 # Size of vocab

embedding_sizes = [300]
#embedding_sizes = [100, 200, 300]  # Dimension of the embedding vector.


"""
Helper functions

"""

# Step 1: Read the data into a list of strings.
def read_data(filename):
    """Extract the file as a list of words."""
    with open(filename) as f:
        data = tf.compat.as_str(f.read()).split()
    
    # Convert to lower case, remove punctuations, non-alphanumeric and stop words
    tokens = [w.lower() for w in data]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    data = [w for w in words if not w in stop_words]    
    return data


def read_sim_dict(filename, length):
    # Read wordsim353 into a dictionary of format: {word: embedding (embedding is emptry for now)}
    sim_dict = {}
    words_list = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] not in words_list:
                words_list.append(row[0])
            if row[1] not in words_list:
                words_list.append(row[1])
    
    for word in words_list:
        sim_dict[word] = np.zeros(length)
    return sim_dict


def read_simcolumn_list(filename, column):
    # Read wordsim353 into a list of format: [word1, word2...]
    sim_list = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sim_list.append(row[column])
    return sim_list
            
# Step 2: Build the dictionary and replace rare words with UNK token.
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


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Fuction to compute cosine similarity
def cos_similarity(a, b):
    if np.linalg.norm(a, ord=2) != 0 and np.linalg.norm(b, ord=2) != 0:
        normalized_a = a / np.linalg.norm(a, ord=2)
        normalized_b = b / np.linalg.norm(b, ord=2)
        similarity = np.dot(normalized_a, normalized_b)
    else:
        similarity = .0
    return similarity


# Step 4: Build and train a skip-gram model.
def train_skipgram(embedding_size):
    batch_size = 128
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 6  # Number of negative examples to sample.
    
    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
          # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
              tf.nn.nce_loss(
                  weights=nce_weights,
                  biases=nce_biases,
                  labels=train_labels,
                  inputs=embed,
                  num_sampled=num_sampled,
                  num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the normalized embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm


        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()
        
    
    log_dir = './log'
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    
    # Step 5: Begin training.
    num_steps = 200001
    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, summary, loss_val = session.run([optimizer, merged, loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                # The average loss is an estimate of the loss over the last 2000
                # batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        final_embeddings = normalized_embeddings.eval()
        
        # Write corresponding labels for the embeddings.
        with open(log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    writer.close()  
    
    for key in sim_d:
        if key in vocabulary and vocabulary.index(key)<=vocabulary_size :
            idx = unused_dictionary.get(key)
#            idx = vocabulary.index(key)
            embed = final_embeddings[idx, :]
            sim_d[key] = embed
            
    return sim_d
    


"""
Actual execution

"""

# Step 1
vocabulary = read_data('wiki/wiki.txt')
print('Data size', len(vocabulary))

# Step 2
# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, unused_dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# Convert simwords into a list of tuples of format ((word1, word2), score)
col1 = read_simcolumn_list('combined.csv', 0)
col2 = read_simcolumn_list('combined.csv', 1)
assert len(col1) == len(col2),'wrong number of words from input'


## Step 3
for embedding_size in embedding_sizes:
    sim_d = read_sim_dict('combined.csv', embedding_size)
    sim_d = train_skipgram(embedding_size)
#    print(sim_d)
    
    with open('output_' + str(embedding_size) + '.csv', mode='w') as similarity_file:
        similarity_writer = csv.writer(similarity_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(col1)):
            word1, word2 = col1[i], col2[i]
            
            if word1 == word2:
                similarity = 1.0
            else:
                embed1, embed2 = sim_d[word1], sim_d[word2]
                similarity = cos_similarity(embed1, embed2)
            similarity_writer.writerow([word1, word2, similarity])

del vocabulary  # to reduce memory.

