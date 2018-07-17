import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


class AE_coord_conv:
    def __init__(self):
        self.name = AE_coord_conv.__name__

    def split_imgs(self, imgs_placeholder):
        """Splits images into base one and target one"""
        base_imgs = imgs_placeholder[:, 0, :, :, :]
        target_imgs = imgs_placeholder[:, -1, :, :, :]
        return base_imgs, target_imgs

    def accumulation_matrix(self, batch_size, dim_size, type):
        """
        Creates matrix filed with increasing numbers, normalized
        :param type: 'vertical' or 'horizontal'
        :return:
        """
        ones = tf.ones([batch_size, dim_size], dtype=tf.float32)
        ones = tf.expand_dims(ones, -1)
        range = tf.tile(tf.expand_dims(tf.range(dim_size), 0), [batch_size, 1])
        range = tf.expand_dims(range, 1)
        channel = tf.matmul(ones, tf.to_float(range))
        channel = tf.expand_dims(channel, -1)
        channel = tf.cast(channel, tf.float32) / (tf.to_float(dim_size) - 1)
        channel = channel * 2 - 1

        if type is 'horizontal':
            return channel
        elif type is 'vertical':
            channel = tf.image.rot90(channel, 3)
            return channel
        else:
            raise Exception('Type must be either horizontal or vertical!')

    def coord_conv(self, input_tensor, filters, kernel, padding, activation):
        # coord maps
        shape = tf.shape(input_tensor)
        batch_size, y_dim, x_dim = shape[0], shape[1], shape[2]
        y_mat = self.accumulation_matrix(batch_size, y_dim, 'vertical')
        x_mat = self.accumulation_matrix(batch_size, x_dim, 'horizontal')
        ret = tf.concat([input_tensor, y_mat, x_mat], axis=-1)
        # conv maps
        conv = tf.layers.conv2d(ret, filters, kernel, padding=padding, activation=activation)
        return conv

    def encoder(self, imgs, activation, is_training):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE, initializer=xavier_initializer(), regularizer=l2_regularizer(0.01)):
            e_conv = self.coord_conv(imgs, 48, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)


            e_conv = self.coord_conv(e_conv, 92, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = self.coord_conv(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = self.coord_conv(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = self.coord_conv(e_conv, 256, 3, padding='same', activation=None)
            e_conv = tf.layers.batch_normalization(e_conv, training=is_training, fused=True)
            e_conv = activation(e_conv)
            e_conv = tf.layers.max_pooling2d(e_conv, 2, 2)

            e_conv = self.coord_conv(e_conv, 256, 3, padding='same', activation=None)
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
