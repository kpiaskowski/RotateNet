import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


class AE_only_conv:
    def __init__(self):
        self.name = AE_only_conv.__name__

    def encoder(self, imgs, activation, is_training):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE, initializer=xavier_initializer(), regularizer=l2_regularizer(0.01)):
            e_conv = tf.layers.conv2d(imgs, 48, 3, padding='same', activation=None, name='econv1')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = tf.layers.conv2d(e_conv, 92, 3, padding='same', activation=None, name='econv2')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = tf.layers.conv2d(e_conv, 256, 3, padding='same', activation=None, name='econv3')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = tf.layers.conv2d(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = tf.layers.conv2d(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = tf.layers.conv2d(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            lv = tf.layers.flatten(e_conv)
            return lv

    def merge_lv_angle(self, lv, angles, activation):
        view = tf.layers.dense(angles, 512)
        view = tf.layers.dense(view, 512, activation)

        concat = tf.concat([lv, view], -1)
        concat = tf.layers.dense(concat, 1024, activation)
        concat = tf.layers.dense(concat, 16384, activation)

        return concat

    def decoder(self, merged_lv, activation, is_training):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE, initializer=xavier_initializer(), regularizer=l2_regularizer(0.01)):
            d_conv = tf.reshape(merged_lv, [-1, 8, 8, 256])
            d_conv = tf.image.resize_images(d_conv, (16, 16))

            d_conv = tf.layers.conv2d(d_conv, 256, 3, padding='same', activation=None)
            d_conv = tf.layers.batch_normalization(d_conv, training=is_training, fused=True)
            d_conv = activation(d_conv)
            d_conv = tf.image.resize_images(d_conv, (32, 32))

            d_conv = tf.layers.conv2d(d_conv, 92, 3, padding='same', activation=None)
            d_conv = tf.layers.batch_normalization(d_conv, training=is_training, fused=True)
            d_conv = activation(d_conv)
            d_conv = tf.image.resize_images(d_conv, (64, 64))

            d_conv = tf.layers.conv2d(d_conv, 48, 3, padding='same', activation=None)
            d_conv = tf.layers.batch_normalization(d_conv, training=is_training, fused=True)
            d_conv = activation(d_conv)
            d_conv = tf.image.resize_images(d_conv, (128, 128))

            d_conv = tf.layers.conv2d(d_conv, 3, 3, padding='same', activation=None)
            return d_conv