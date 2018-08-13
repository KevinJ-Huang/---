# Copyright 2016 Google Inc.
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
"""Useful image metrics."""

import tensorflow as tf
import utils
import vgg
import models
from utils import Getfilter,blur
PATCH_WIDTH = 100
PATCH_HEIGHT = 100
w_content = 10
w_texture = 1
w_color = 0.8
w_filter = 8
w_tv = 2000

import numpy as np

# content loss
def content_loss(target, prediction,batch_size):
  CONTENT_LAYER = 'relu5_4'
  vgg_dir = '/gdata/huangjie/hdrnet/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
  enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(prediction * 255))
  dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(target * 255))

  content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
  loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
  return tf.reduce_mean(loss_content)

#texture loss
def filter_loss(target, prediction, adv_):
  prediction = tf.image.rgb_to_grayscale(Getfilter(19,prediction))
  target = tf.image.rgb_to_grayscale(Getfilter(19,target))
  enhanced = tf.reshape(prediction, [-1, PATCH_WIDTH * PATCH_HEIGHT])
  dslr = tf.reshape(target, [-1, PATCH_WIDTH * PATCH_HEIGHT])
  adversarial_ = tf.multiply(enhanced, 1 - adv_) + tf.multiply(dslr, adv_)
  adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
  discrim_predictions = models.adversarial(adversarial_image)
  discrim_target = tf.concat([adv_, 1 - adv_], 1)

  loss_filter= -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
  correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
  discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  return -loss_filter, discim_accuracy


def texture_loss(target,prediction,adv_1):
  prediction1 = tf.image.rgb_to_grayscale(blur(19,prediction))
  target1 = tf.image.rgb_to_grayscale(blur(19,target))
  enhanced1 = tf.reshape(prediction1, [-1, PATCH_WIDTH * PATCH_HEIGHT])
  dslr1 = tf.reshape(target1, [-1, PATCH_WIDTH * PATCH_HEIGHT])
  adversarial_1 = tf.multiply(enhanced1, 1 - adv_1) + tf.multiply(dslr1, adv_1)
  adversarial_image1 = tf.reshape(adversarial_1, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
  discrim_predictions1 = models.adversarial_1(adversarial_image1)
  discrim_target1 = tf.concat([adv_1, 1 - adv_1], 1)

  loss_texture = -tf.reduce_sum(discrim_target1 * tf.log(tf.clip_by_value(discrim_predictions1, 1e-10, 1.0)))
  correct_predictions1 = tf.equal(tf.argmax(discrim_predictions1, 1), tf.argmax(discrim_target1, 1))
  discim_accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, tf.float32))
  return -loss_texture, discim_accuracy1


#color loss
def color_loss(target,prediction,batch_size):
  loss_color = tf.reduce_sum(tf.pow(target - prediction,2))/(2*batch_size)
  return loss_color


def tv_loss(prediction,batch_size):
  batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
  tv_y_size = utils._tensor_size(prediction[:, 1:, :, :])
  tv_x_size = utils._tensor_size(prediction[:, :, 1:, :])
  y_tv = tf.nn.l2_loss(prediction[:, 1:, :, :] - prediction[:, :batch_shape[1] - 1, :, :])
  x_tv = tf.nn.l2_loss(prediction[:, :, 1:, :] - prediction[:, :, :batch_shape[2] - 1, :])
  loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
  return loss_tv



def l2_loss(target, prediction, adv_, adv_1, batch_size,name=None):
  with tf.name_scope(name, default_name='l2_loss', values=[target, prediction]):
    # loss = tf.reduce_mean(tf.square(target-prediction))
    loss_content = content_loss(target,prediction,batch_size)
    loss_filter, discim_accuracy = filter_loss(target,prediction,adv_)
    loss_color = color_loss(target,prediction,batch_size)
    loss_tv = tv_loss(prediction,batch_size)
    loss_texture, discim_accuracy1 = texture_loss(target,prediction,adv_1)
    loss = w_content * loss_content + w_color * loss_color + w_filter * loss_filter + w_texture * loss_texture + w_tv * loss_tv
  return loss, loss_content, loss_color, loss_filter, loss_texture, loss_tv, discim_accuracy, discim_accuracy1


def psnr(target, prediction, name=None):
  with tf.name_scope(name, default_name='psnr_op', values=[target, prediction]):
    squares = tf.square(target-prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = tf.reduce_mean((-10/np.log(10))*tf.log(tf.reduce_mean(squares, axis=[1])))
  return p



