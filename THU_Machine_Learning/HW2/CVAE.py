#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import gzip

import tensorflow as tf
import six
from six.moves import range
from six.moves import cPickle as pickle
import numpy as np
from skimage import io, img_as_ubyte
import zhusuan as zs
from keras.utils import to_categorical


def save_image_collections(x, filename, shape=(10, 10)):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def load_mnist_realval(path):
    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    n_y = t_train.max() + 1
    return x_train, t_train, x_valid, t_valid, x_test, t_test


# p(x|z) -> decoder net
@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(y, x_dim, z_dim, y_dim, n):
    bn = zs.BayesianNet()
    # for sampling z ~ q(z|x); assume p(z) = N(0,1)
    z = bn.normal("z", tf.zeros([n, z_dim]), std=1., group_ndims=1)
    # concatenate z and y
    z = tf.concat(axis=1, values=[z, y])  
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    x_mean = bn.deterministic("x_mean", tf.sigmoid(x_logits))
    # add noise, x ~ Bernoulli(x_logits)
    x = bn.bernoulli("x", x_logits, group_ndims=1, dtype=tf.float32)
    return bn

# q(z|x) -> encoder net
@zs.reuse_variables(scope="q_net")
def build_q_net(x, y, z_dim, y_dim):
    bn = zs.BayesianNet()
    # concatenate x and y
    x = tf.concat(axis=1, values=[x, y])
    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1)
    return bn


def build_train(meta_model, variational, x):
    # shape: [batch_size]
    # compute ELBO
    lower_bound = zs.variational.elbo(meta_model, {"x": x}, variational=variational)
    # reparameterize with sgvb() method
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)
    return infer_op, lower_bound


def random_generation(meta_model):
    x_gen = tf.reshape(meta_model.observe()["x_mean"], [-1, 28, 28, 1])
    return x_gen


def main():
    # Load MNIST, x_train shape: [n_train, 28 * 28 * 1]
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval("mnist.pkl.gz")
    
    # 
    x_train = np.random.binomial(1, x_train, size=x_train.shape).astype(np.float32)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype(np.float32)
    x_dim = x_train.shape[1]
    
    y_train = to_categorical(np.array(t_train))
    y_test = to_categorical(np.array(t_test))
    y_dim = y_train.shape[1]
    y_pic = to_categorical(np.arange(10))
    
    # Define model parameters
    z_dim = 40

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    meta_model = build_gen(y, x_dim, z_dim, y_dim, n)
    variational = build_q_net(x, y, z_dim, y_dim)

    infer_op, lower_bound = build_train(meta_model, variational, x)

    # Random generation
    x_gen = random_generation(meta_model)

    # Define training/evaluation parameters
    epochs = 1
    batch_size = 128
    iters = x_train.shape[0] // batch_size

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x: x_batch,
                                            y: y_batch,
                                            n: batch_size})
                lbs.append(lb)
            print("Epoch {}: Lower bound = {}".format(epoch, np.mean(lbs)))
            
            images = sess.run(x_gen, feed_dict={y: y_pic, n: 10})
            name = os.path.join("results", "vae.epoch.{}.png".format(epoch))
            save_image_collections(images, name, shape=(1, 10))


if __name__ == "__main__":
    main()
