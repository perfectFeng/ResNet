#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import random
import sys
import os
import event_read
import resnet
import brnn_model
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"

# model_filename = "chckPts/resnet_v1_50.ckpt"
model_filename1 = "chckPts/save4040.ckpt"
start_step = 0

batch_size = resnet.batch_size
num_frames = resnet.num_frames
height = resnet.height
width = resnet.width
channels = resnet.channels
n_classes = resnet.NUM_CLASSES
hidden_size = brnn_model.hidden_size

max_iters = 8


def _variable_with_weight_decay(name, shape, wd):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def calc_reward(logit):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit)
    )

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('total_loss', loss_)
    return loss_


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def evaluate():
    nextX, nextY = event_read.readTestFile(batch_size, num_frames)
    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
    r = sess.run(accuracy, feed_dict=feed_dict)

    print("ACCURACY: " + str(r))


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        labels_placeholder = tf.placeholder(tf.int64, shape=batch_size)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, channels))

        with tf.variable_scope('b_rnn'):
            rw = _variable_with_weight_decay('rw', [hidden_size, n_classes], 0.0005)
            rb = _variable_with_weight_decay('rb', [n_classes], 0.000)

        feature = resnet.inference(inputs_placeholder)
        outputs = brnn_model.BiRNN(feature, rw, rb)

        loss = calc_reward(outputs)

        param = tf.trainable_variables()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, var_list=param)

        accuracy = tower_acc(outputs, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # init = tf.global_variables_initializer()
            # sess.run(init)
            sess.graph.finalize()
            '''
            variables = tf.contrib.framework.get_variables_to_restore(include=['b_rnn/rw', 'b_rnn/rb'])
            restore = tf.train.Saver(variables)
            restore.restore(sess, model_filename1)
            variables1 = tf.contrib.framework.get_variables_to_restore(
                include=['resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights',
                         'resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights',
                         'resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights',
                         'resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights',
                         'resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights',
                         'resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights',
                         'resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights',
                         'resnet_v1_50/conv1/weights',
                         'resnet_v1_50/logits/weights',
                         'resnet_v1_50/logits/biases'])
            restore1 = tf.train.Saver(variables1)
            restore1.restore(sess, model_filename)
			'''
            saver.restore(sess, model_filename1)

            summary_writer = tf.summary.FileWriter(summaryFolderName, graph=sess.graph)
            # training
            for epoch in range(max_iters):

                lines = event_read.readFile()

                for batch in range(int(len(lines) / batch_size)):

                    start_time = time.time()
                    nextX, nextY = event_read.readTrainData(batch, lines, batch_size, num_frames)

                    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}

                    _, summary, l, acc = sess.run([train_op, merged, loss, accuracy], feed_dict=feed_dict)

                    duration = time.time() - start_time

                    print('epoch-step %d-%d: %.3f sec' % (epoch, batch, duration))

                    if batch % 10 == 0:
                        saver.save(sess,
                                   save_dir + save_prefix + str(epoch * int(len(lines) / batch_size) + batch) + ".ckpt")
                        print('loss:', l, '---', 'acc:', acc)
                        summary_writer.add_summary(summary, epoch * int(len(lines) / batch_size) + batch)
                        evaluate()
