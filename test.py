#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import random
import sys
import os
import test_event_read as read
import resnet
import brnn_model
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = resnet.batch_size
num_frames = resnet.num_frames
height = resnet.height
width = resnet.width
channels = resnet.channels
n_classes = resnet.NUM_CLASSES
hidden_size = brnn_model.hidden_size
s = read.segment

model_filename = "./chckPts/save10115.ckpt"


def _variable_with_weight_decay(name, shape, wd):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    return var


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, channels))

        with tf.variable_scope('b_rnn'):
            rw = _variable_with_weight_decay('rw', [hidden_size, n_classes], 0.0005)
            rb = _variable_with_weight_decay('rb', [n_classes], 0.000)

        feature = resnet.inference(inputs_placeholder)
        outputs = brnn_model.BiRNN(feature, rw, rb)
        outputs = tf.nn.softmax(outputs)
        predict = []

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_filename)

            # test
            videos = read.readFile()
            for l in videos:

                for batch in range(int(len(l[1]) / batch_size)):

                        nextX, segments = read.readTrainData(batch, l, batch_size)

                        feed_dict = {inputs_placeholder: nextX}

                        output = sess.run(outputs, feed_dict=feed_dict)
                        for i in range(batch_size):
                            p_label = np.argmax(output[i])
                            if p_label != 0:
                                p = segments[i][0] + ' ' + str(segments[i][1]) + ' ' + str(segments[i][2]) + ' ' + \
                                    str(p_label) + ' ' + str(output[i][p_label])
                                p = p+'\n'
                                write_file = open(l[0].split('.')[0] + "_" + str(s) + ".txt", "a")
                                write_file.write(p)
                                write_file.close()

