from architectures.AE_only_conv import AE_only_conv
from dataproviders.chairs_provider import ChairProvider
import tensorflow as tf
import datetime
from utils.functions import normalize_inputs, cropping_pipeline
import os

batch_size = 25
learning_rate = 0.0005
n_imgs = 2
img_size = 128
iters = 50000
ckpt = 20
save_ckpt = 2000
activation = tf.nn.relu
note = 'cropped images'

dataprovider = ChairProvider('../RotateNet_data/chairs', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
handle, t_iter, v_iter, images, angles, classes = dataprovider.dataset()

model = AE_only_conv()
model_name = 'chairs|--|{}|--|batch_{}|--|eta_{}|--|imgsize_{}|--|activ_{}|--|nimgs_{}|--|date_{}|--|{}'.format(model.name,
                                                                                                                batch_size,
                                                                                                                learning_rate,
                                                                                                                img_size,
                                                                                                                activation.__name__,
                                                                                                                n_imgs,
                                                                                                                datetime.datetime.now(),
                                                                                                                note)

normalized_imgs, normalized_angles = normalize_inputs(images, angles)
base_imgs, target_imgs = model.split_imgs(normalized_imgs)
target_angles = normalized_angles[:, -1, :]
reshaped_angles = tf.reshape(target_angles, [-1, 3])  # tf hack

base_masks = 1 - tf.to_float(tf.greater_equal(base_imgs, 0.95))[..., :1]  # preserve dims
target_masks = 1 - tf.to_float(tf.greater_equal(target_imgs, 0.95))[..., :1]  # preserve dims


# placeholders
is_training = tf.placeholder(tf.bool)
base_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
target_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])

lv = model.encoder(base_pl, activation, is_training)
merged_lv = model.merge_lv_angle(lv, reshaped_angles, activation)
gen_imgs = model.decoder(merged_lv, activation, is_training)

# losses
mse_loss = tf.losses.mean_squared_error(labels=target_pl, predictions=gen_imgs)

# summaries
concat_img = tf.concat([base_pl, gen_imgs, target_pl], 2)
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

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/' + model_name + '_train')
    val_writer = tf.summary.FileWriter('summaries/' + model_name + '_val')
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    t_s, v_s = 0, 0
    for i in range(iters):
        base, bmasks, target, tmasks = sess.run([base_imgs, base_masks, target_imgs, target_masks], feed_dict={handle: t_handle})
        resized_base = cropping_pipeline(base, bmasks, img_size)
        resized_target = cropping_pipeline(target, tmasks, img_size)
        _, cost, summ = sess.run([train_op, mse_loss, loss_merged], feed_dict={base_pl: resized_base, target_pl: resized_target, handle: t_handle, is_training: True})
        print('TRAIN iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
        train_writer.add_summary(summ, t_s)
        train_writer.flush()
        t_s += 1

        if t_s % ckpt == 0:
            base, bmasks, target, tmasks = sess.run([base_imgs, base_masks, target_imgs, target_masks], feed_dict={handle: t_handle})
            resized_base = cropping_pipeline(base, bmasks, img_size)
            resized_target = cropping_pipeline(target, tmasks, img_size)
            img_summ = sess.run(img_merged, feed_dict={base_pl: resized_base, target_pl: resized_target, handle: t_handle, is_training: False})
            # img_summ = sess.run(img_merged, feed_dict={handle: t_handle, is_training: False})
            train_writer.add_summary(img_summ, t_s)
            train_writer.flush()
            t_s += 1

            base, bmasks, target, tmasks = sess.run([base_imgs, base_masks, target_imgs, target_masks], feed_dict={handle: v_handle})
            resized_base = cropping_pipeline(base, bmasks, img_size)
            resized_target = cropping_pipeline(target, tmasks, img_size)
            img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={base_pl: resized_base, target_pl: resized_target, handle: v_handle, is_training: False})
            # img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={handle: v_handle, is_training: False})
            print('VAL iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
            val_writer.add_summary(img_summ, v_s)
            val_writer.add_summary(loss_summ, v_s)
            val_writer.flush()
            v_s += 1

        if t_s % save_ckpt == 0:
            if not os.path.isdir('saved_models/' + model_name):
                os.mkdir('saved_models/' + model_name)
            saver.save(sess, 'saved_models/' + model_name + '/' + 'model.ckpt', t_s)
            print('Model saved at {} step'.format(t_s))