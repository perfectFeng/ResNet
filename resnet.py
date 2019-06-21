import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

NUM_CLASSES = 12

height = 224
width = 224
batch_size = 3
num_frames = 32
channels = 3


def inference(x, use_bias=False):

    x = tf.reshape(x, [batch_size * num_frames, height, width, channels])

    with tf.variable_scope('resnet_v1_50'):

        num_blocks = [3, 4, 6, 3]

        c = Config()

        c['ksize'] = 3
        c['stride'] = 1
        c['use_bias'] = use_bias
        c['stack_stride'] = 2

        with tf.variable_scope('conv1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = conv(x, c)
            x = bn(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('block1'):
            x = _max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64
            x = stack(x, c)

        with tf.variable_scope('block2'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            assert c['stack_stride'] == 2
            x = stack(x, c)

        with tf.variable_scope('block3'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = stack(x, c)

        with tf.variable_scope('block4'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = stack(x, c)

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if NUM_CLASSES is not None:
            with tf.variable_scope('logits'):
                x = fc(x)

    return x


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('unit_%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']
    with tf.variable_scope('bottleneck_v1'):
        with tf.variable_scope('conv1'):
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                x = conv(x, c)
                x = bn(x, c)
                x = tf.nn.relu(x)

        with tf.variable_scope('conv2'):
                x = conv(x, c)
                x = bn(x, c)
                x = tf.nn.relu(x)

        with tf.variable_scope('conv3'):
                c['conv_filters_out'] = filters_out
                c['ksize'] = 1
                assert c['stride'] == 1
                x = conv(x, c)
                x = bn(x, c)

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                shortcut = conv(shortcut, c)
                shortcut = bn(shortcut, c)

    return tf.nn.relu(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                                 initializer=tf.zeros_initializer)
        return x + bias
    with tf.variable_scope('BatchNorm'):
        x = tf.contrib.layers.batch_norm(x, scope='batch_norm', is_training=True)

    return x


def fc(x):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=0.01)

    weights = _get_variable('weights',
                            shape=[1, 1, num_units_in, 1000],
                            initializer=weights_initializer,
                            weight_decay=0.00004)
    biases = _get_variable('biases',
                           shape=[1000],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, tf.reshape(weights, [2048, 1000]), biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, 'resnet_variables']
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    weights = _get_variable('weights',
                                shape=shape,
                                dtype='float',
                                initializer=initializer,
                                weight_decay=0.00004)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
