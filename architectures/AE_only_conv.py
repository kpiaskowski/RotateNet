import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


class AE_only_conv:
    def __init__(self):
        self.name = AE_only_conv.__name__

    def encoder(self, imgs, activation, is_training, batch_size):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE, initializer=xavier_initializer(), regularizer=l2_regularizer(0.01)):
            e_conv = tf.layers.conv2d(imgs, 48, 3, padding='same', activation=None, name='econv1')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            shortcut1 = tf.get_variable('shortcut1', shape=[1, 64, 64, 48], dtype=tf.float32)
            shortcut1 = tf.tile(shortcut1, [batch_size, 1, 1, 1])
            shortcut1 = tf.nn.sigmoid(e_conv * shortcut1)

            e_conv = tf.layers.conv2d(e_conv, 92, 3, padding='same', activation=None, name='econv2')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            shortcut2 = tf.get_variable('shortcut2', shape=[1, 32, 32, 92], dtype=tf.float32)
            shortcut2 = tf.tile(shortcut2, [batch_size, 1, 1, 1])
            shortcut2 = tf.nn.sigmoid(e_conv * shortcut2)

            e_conv = tf.layers.conv2d(e_conv, 256, 3, padding='same', activation=None, name='econv3')
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            shortcut3 = tf.get_variable('shortcut3', shape=[1, 16, 16, 256], dtype=tf.float32)
            shortcut3 = tf.tile(shortcut3, [batch_size, 1, 1, 1])
            shortcut3 = tf.nn.sigmoid(e_conv * shortcut3)

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
            return lv, shortcut1, shortcut2, shortcut3

    def merge_lv_angle(self, lv, angles, activation):
        view = tf.layers.dense(angles, 512)
        view = tf.layers.dense(view, 512, activation)

        concat = tf.concat([lv, view], -1)
        concat = tf.layers.dense(concat, 1024, activation)
        concat = tf.layers.dense(concat, 16384, activation)

        return concat

    def decoder(self, merged_lv, activation, is_training, ag_1, ag_2, ag_3):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE, initializer=xavier_initializer(), regularizer=l2_regularizer(0.01)):
            conv_core = tf.reshape(merged_lv, [-1, 8, 8, 256])
            conv_core = tf.image.resize_images(conv_core, (16, 16))
            conv_core = tf.concat([conv_core, ag_3], axis=-1)

            conv_core = tf.layers.conv2d(conv_core, 256, 3, padding='same', activation=None)
            conv_core = tf.layers.batch_normalization(conv_core, training=is_training, fused=True)
            conv_core = activation(conv_core)
            conv_core = tf.image.resize_images(conv_core, (32, 32))
            conv_core = tf.concat([conv_core, ag_2], axis=-1)

            rgb_conv = tf.layers.conv2d(conv_core, 92, 3, padding='same', activation=None)
            rgb_conv = tf.layers.batch_normalization(rgb_conv, training=is_training, fused=True)
            rgb_conv = activation(rgb_conv)
            rgb_conv = tf.image.resize_images(rgb_conv, (64, 64))
            rgb_conv = tf.concat([rgb_conv, ag_1], axis=-1)

            rgb_conv = tf.layers.conv2d(rgb_conv, 48, 3, padding='same', activation=None)
            rgb_conv = tf.layers.batch_normalization(rgb_conv, training=is_training, fused=True)
            rgb_conv = activation(rgb_conv)
            rgb_conv = tf.image.resize_images(rgb_conv, (128, 128))

            rgb_conv = tf.layers.conv2d(rgb_conv, 3, 3, padding='same', activation=None)

            depth_conv = tf.layers.conv2d(conv_core, 92, 3, padding='same', activation=None)
            depth_conv = tf.layers.batch_normalization(depth_conv, training=is_training, fused=True)
            depth_conv = activation(depth_conv)
            depth_conv = tf.image.resize_images(depth_conv, (64, 64))
            depth_conv = tf.concat([depth_conv, ag_1], axis=-1)

            depth_conv = tf.layers.conv2d(depth_conv, 48, 3, padding='same', activation=None)
            depth_conv = tf.layers.batch_normalization(depth_conv, training=is_training, fused=True)
            depth_conv = activation(depth_conv)
            depth_conv = tf.image.resize_images(depth_conv, (128, 128))

            depth_conv = tf.layers.conv2d(depth_conv, 1, 3, padding='same', activation=None)

            return rgb_conv, depth_conv
