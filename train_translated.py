import cv2

from architectures.AE_coord_conv import AE_coord_conv
from dataproviders.chairs_provider import ChairProvider
import tensorflow as tf
import datetime

from utils.functions import normalize_inputs
import os

batch_size = 25
learning_rate = 0.0005
n_imgs = 5
img_size = 128
iters = 50000
ckpt = 20
save_ckpt = 2000
activation = tf.nn.relu
note = 'rolled matrices with coord conv +A+E'
max_roll = 35

# dataprovider
dataprovider = ChairProvider('../RotateNet_data/chairs', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
handle, t_iter, v_iter, images, angles, classes = dataprovider.dataset()

# naming
model = AE_coord_conv()
model_name = 'chairs|--|{}|--|batch_{}|--|eta_{}|--|imgsize_{}|--|activ_{}|--|nimgs_{}|--|date_{}|--|{}'.format(model.name,
                                                                                                                batch_size,
                                                                                                                learning_rate,
                                                                                                                img_size,
                                                                                                                activation.__name__,
                                                                                                                n_imgs,
                                                                                                                datetime.datetime.now(),
                                                                                                                note)

# # placeholders
is_training = tf.placeholder(tf.bool)

# data
normalized_imgs, normalized_angles = normalize_inputs(images, angles)
base_imgs, target_imgs = model.split_imgs(normalized_imgs)

# rolling
x_roll = tf.random_uniform([1], -max_roll, max_roll, tf.int32)[0]
y_roll = tf.random_uniform([1], -max_roll, max_roll, tf.int32)[0]
rolled_b = tf.manip.roll(base_imgs, y_roll, axis=1)
rolled_b = tf.manip.roll(rolled_b, x_roll, axis=2)
rolled_t = tf.manip.roll(target_imgs, y_roll, axis=1)
rolled_t = tf.manip.roll(rolled_t, x_roll, axis=2)

# masking
masks = 1 - tf.to_float(tf.equal(rolled_b, 1))[..., :1]  # preserve dims
concat_base_imgs = tf.concat([rolled_b, masks], axis=-1)

target_angles = normalized_angles[:, -1, :]

# model
reshaped_imgs = tf.reshape(concat_base_imgs, [-1, img_size, img_size, 4])  # tf hack
reshaped_angles = tf.reshape(target_angles, [-1, 3])  # tf hack
lv = model.encoder(reshaped_imgs, activation, is_training)
merged_lv = model.merge_lv_angle(lv, reshaped_angles, activation)
gen_imgs = model.decoder(merged_lv, activation, is_training)

# losses
mse_loss = tf.losses.mean_squared_error(labels=rolled_t, predictions=gen_imgs)

# summaries
concat_img = tf.concat([rolled_b, gen_imgs, rolled_t], 2)
loss_summary = tf.summary.scalar('loss', mse_loss)
img_summary = tf.summary.image('images', concat_img)
loss_merged = tf.summary.merge([loss_summary])
img_merged = tf.summary.merge([img_summary])

# train ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(mse_loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/' + model_name + '_train')
    val_writer = tf.summary.FileWriter('summaries/' + model_name + '_val')
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    t_s, v_s = 0, 0
    for i in range(iters):
        _, cost, summ = sess.run([train_op, mse_loss, loss_merged], feed_dict={handle: t_handle, is_training: True})
        print('TRAIN iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
        train_writer.add_summary(summ, t_s)
        train_writer.flush()
        t_s += 1

        if t_s % save_ckpt == 0:
            if not os.path.isdir('saved_models/' + model_name):
                os.mkdir('saved_models/' + model_name)
            saver.save(sess, 'saved_models/' + model_name + '/' + 'model.ckpt', t_s)
            print('Model saved at {} step'.format(t_s))

        if t_s % ckpt == 0:
            img_summ = sess.run(img_merged, feed_dict={handle: t_handle, is_training: False})
            train_writer.add_summary(img_summ, t_s)
            train_writer.flush()
            t_s += 1

            img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={handle: v_handle, is_training: False})
            print('VAL iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
            val_writer.add_summary(img_summ, v_s)
            val_writer.add_summary(loss_summ, v_s)
            val_writer.flush()
            v_s += 1
