import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import upsample_layer

def Resnet(img_low,img_high,img_ori):

    with tf.variable_scope("generator"):


        rgb_prediction = RGBmodel(img_low)
        up_prediction = upsample_layer(rgb_prediction)
        high_prediction = Filtermodel(img_high)


        # quanzhi add method
        # weights = tf.get_variable('weights',shape=[1],dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # weights1 = tf.get_variable('weights1',shape=[1],dtype=tf.float32,
        #                            initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # conv=tf.multiply(weigredhts,up_prediction)
        # conv1=tf.multiply(weights1,high_prediction)


        #group cnn
        red = tf.concat([up_prediction[:,:,:,:1],high_prediction[:,:,:,:1]],axis=3)
        green = tf.concat([up_prediction[:,:,:,1:2],high_prediction[:,:,:,1:2]],axis=3)
        blue = tf.concat([up_prediction[:,:,:,2:3],high_prediction[:,:,:,2:3]],axis=3)

        W1 = weight_variable([3, 3, 2, 1], name="W1"); b1 = bias_variable([1], name="b1")
        R = tf.nn.relu(conv2d(red, W1) + b1)
        W2 = weight_variable([3, 3, 2, 1], name="W2"); b2 = bias_variable([1], name="b2")
        G = tf.nn.relu(conv2d(green, W2) + b2)
        W3 = weight_variable([3, 3, 2, 1], name="W3"); b3 = bias_variable([1], name="b3")
        B = tf.nn.relu(conv2d(blue, W3) + b3)

        img_change = tf.concat([B[:,:,:,:1],G[:,:,:,:1],R[:,:,:,:1]],axis=3)
        img_changed = tf.add(img_change, img_ori)


        enhance = Changemodel(img_changed)
        
    return enhance


def RGBmodel(input_image):

    with tf.variable_scope("generator_rgb"):

        W1 = weight_variable([3, 3, 3, 64], name="W1"); b1 = bias_variable([64], name="b1")
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2")
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3")
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3))+c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4")
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))


        W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5")
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5))+c3

        W6 = weight_variable([3, 3, 64, 64], name="W6"); b6 = bias_variable([64], name="b6")
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7"); b7 = bias_variable([64], name="b7")
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7))+c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8"); b8 = bias_variable([64], name="b8")
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9"); b9 = bias_variable([64], name="b9")
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9))+c7

        # residual 5

        W10 = weight_variable([3, 3, 64, 64], name="W10"); b10 = bias_variable([64], name="b10")
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11"); b11 = bias_variable([64], name="b11")
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)+c9

        # Final

        W12 = weight_variable([3, 3, 64, 3], name="W12");b12 = bias_variable([3], name="b12")
        enhanced_rgb = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced_rgb




def Filtermodel(input_image):

    with tf.variable_scope("generator_filter"):

        W1 = weight_variable([3, 3, 3, 64], name="W1"); b1 = bias_variable([64], name="b1")
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2")
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3")
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3))+c1

        # residual 2
        W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4")
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))


        W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5")
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5))+c3

        W6 = weight_variable([3, 3, 64, 3], name="W6"); b6 = bias_variable([3], name="b6")
        enhanced_filter = tf.nn.tanh(conv2d(c5, W6) + b6) * 0.58 + 0.5

    return enhanced_filter




def Changemodel(input_image):

    with tf.variable_scope("generator_change"):

        W1 = weight_variable([3, 3, 3, 64], name="W1"); b1 = bias_variable([64], name="b1")
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2")
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3")
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1


        W4 = weight_variable([3, 3, 64, 3], name="W4"); b4 = bias_variable([3], name="b4")
        enhanced_change = tf.nn.tanh(conv2d(c3, W4) + b4) * 0.58 + 0.5

    return enhanced_change



def adversarial(image_):
  with tf.variable_scope("discriminator"):
    conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)
    conv2 = _conv_layer(conv1, 128, 5, 2)
    conv3 = _conv_layer(conv2, 192, 3, 1)
    conv4 = _conv_layer(conv3, 192, 3, 1)
    conv5 = _conv_layer(conv4, 128, 3, 2)

    flat_size = 128 * 7 * 7
    conv5_flat = tf.reshape(conv5, [-1, flat_size])

    W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
    bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

    fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

    W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
    bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

    adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)

  return adv_out


def adversarial_1(image_):
  with tf.variable_scope("discriminator1"):
    conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)
    conv2 = _conv_layer(conv1, 128, 5, 2)
    conv3 = _conv_layer(conv2, 192, 3, 1)
    conv4 = _conv_layer(conv3, 192, 3, 1)
    conv5 = _conv_layer(conv4, 128, 3, 2)

    flat_size = 128 * 7 * 7
    conv5_flat = tf.reshape(conv5, [-1, flat_size])

    W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
    bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

    fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

    W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
    bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

    adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)

  return adv_out


def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
  weights_init = _conv_init_vars(net, num_filters, filter_size)
  strides_shape = [1, strides, strides, 1]
  bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

  net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias
  net = leaky_relu(net)

  if batch_nn:
    net = _instance_norm(net)

  return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)
