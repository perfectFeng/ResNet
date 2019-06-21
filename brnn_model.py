#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

n_input = 1000
n_step = 32
hidden_size = 256


def BiRNN(x, w, b):

    x = tf.reshape(x, [-1, n_step, n_input])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_step)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,
                                             forget_bias=1.0,
                                             state_is_tuple=True)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    '''
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,
                                                forget_bias = 1.0,
                                                state_is_tuple = True)
    
    output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        lstm_fw_cell,
        lstm_bw_cell,
        x,
        dtype = tf.float32
        )'''

    return tf.matmul(outputs[-1], w) + b
