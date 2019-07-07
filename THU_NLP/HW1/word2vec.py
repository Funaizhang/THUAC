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

from utils import *

import collections
import math
import os
import random
import csv
from tempfile import gettempdir
import numpy as np 
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from scipy.stats import spearmanr

    
"""
Main script

"""
class word2vec():
    
    def __init__ (self, train_filename, eval_filename, embedding_size = 300, num_steps = 2000001, num_sampled = 6):
        # training_data takes in a list of strings
        # eval_data takes in df object
        # num_sampled refers to number of negative examples to sample.
        
        self.train_filename = train_filename
        self.eval_filename = eval_filename
        self.embedding_size = embedding_size
        self.num_steps = num_steps
        self.num_sampled = num_sampled
        self.vocabulary_size = 50000
        self.batch_size = 128
        self.skip_window = 1 # How many words to consider left and right.
        self.num_skips = 2 # How many times to reuse an input to generate a label.
        self.data_index = 0
        self.log_dir = './log'
        
        
        vocabulary = read_data(self.train_filename)
        # Filling 4 global variables:
        # data - list of codes (integers from 0 to vocabulary_size-1). This is the original text but words are replaced by their codes
        # count - map of words(strings) to count of occurrences
        # dictionary - map of words(strings) to their codes(integers)
        # reverse_dictionary - maps codes(integers) to words(strings)
        self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(vocabulary, self.vocabulary_size)
        print('Most common words (+UNK)', self.count[:5])
        print('Sample data', self.data[:10], [self.reverse_dictionary[i] for i in self.data[:10]])
        del vocabulary
        
        # Convert simwords into a list of tuples of format [(word1, word2)] and [score]
        # test_words Only contains valid words
        self.test_words, self.human_scores = read_sim_list(self.eval_filename, self.dictionary)
 
    
    # Function to generate a training batch for the skip-gram model.
    def generate_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels
    
    
    # Build and train a skip-gram model.
    def train_skipgram(self):
        
        loss_list = []
        spearman_list = []
        
        graph = tf.Graph()
        with graph.as_default():
    
            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                self.test_inputs = tf.placeholder(tf.int32, shape=[None, 2])
    
            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
              # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                # Construct the variables for the NCE loss
                with tf.name_scope('weights'):
                    nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
                with tf.name_scope('biases'):
                    nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
    
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
            # Explanation of the meaning of NCE loss: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                  tf.nn.nce_loss(
                      weights=nce_weights,
                      biases=nce_biases,
                      labels=train_labels,
                      inputs=embed,
                      num_sampled=self.num_sampled,
                      num_classes=self.vocabulary_size))
    
            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)
    
            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
            # Compute the normalized embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
    
            # Compute cosine similarity
            test_embeddings1 = tf.nn.embedding_lookup(normalized_embeddings, self.test_inputs[:, 0])
            test_embeddings2 = tf.nn.embedding_lookup(normalized_embeddings, self.test_inputs[:, 1])
            self.cos_similarity = tf.reduce_sum(tf.multiply(test_embeddings1, test_embeddings2), axis=1)
            
        
            # Merge all summaries.
            merged = tf.summary.merge_all()
    
            # Add variable initializer.
            init = tf.global_variables_initializer()
    
            # Create a saver.
            saver = tf.train.Saver()
            
        
        # Create the directory for TensorBoard variables if there is not.
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        
        # Begin training. 
        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(self.log_dir, session.graph)
    
            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')
    
            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels = self.generate_batch(self.batch_size, self.num_skips, self.skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
                # Define metadata variable.
                run_metadata = tf.RunMetadata()
    
                # We perform one update step by evaluating the optimizer op (including it in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable. Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run([optimizer, merged, loss], feed_dict=feed_dict, run_metadata=run_metadata)
                average_loss += loss_val
    
                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (self.num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)
    
                if step % 10000 == 0:
                    if step > 0:
                        average_loss /= 10000
                    # The average loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    loss_list.append(average_loss)
                    average_loss = 0
                
                    self.test_score = session.run(self.cos_similarity, feed_dict={self.test_inputs: self.test_words})
                    correlation = spearmanr(self.human_scores, self.test_score)
                    print('Spearman correlation: {}'.format(correlation))
                    spearman_list.append(correlation)
    
            final_embeddings = normalized_embeddings.eval()
            print(final_embeddings)
            
            # Write corresponding labels for the embeddings.
            with open(self.log_dir + '/metadata.tsv', 'w') as f:
                for i in xrange(self.vocabulary_size):
                    f.write(self.reverse_dictionary[i] + '\n')
    
            # Save the model for checkpoints.
            saver.save(session, os.path.join(self.log_dir, 'model.ckpt'))
    
        writer.close()  
        
#        print(loss_list)
#        print(spearman_list)
        

"""
Actual execution

"""

embedding_sizes = [100, 200, 300]  # Dimension of the embedding vector.

for embedding_size in embedding_sizes:
    model = word2vec('./wiki_corpus/xaa','./wordsim353/combined.csv', embedding_size)
    model.train_skipgram()

