from architectures.AE_only_conv import AE_only_conv
from dataproviders.shapenet_provider import ShapenetProvider
import tensorflow as tf
import datetime

from utils.functions import cropping_pipeline, normalize_images, split_imgs, deg2rad
import os

batch_size = 40
learning_rate = 0.0005
n_imgs = 2
img_size = 128
iters = 500000
ckpt = 20
save_ckpt = 10000
activation = tf.nn.relu
mean_img_val = 0.55
note = 'shapenet attention cropped, not cosined, masked, relative angles, only mse_loss 2'

# dataprovider
dataprovider = ShapenetProvider('../shapenet', '../shapenet_raw', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
handle, t_iter, v_iter, images, masks, angles, classes = dataprovider.dataset()

# namingyou
model = AE_only_conv()
model_name = '{}---date_{}'.format(note, datetime.datetime.now())

# placeholders
is_training = tf.placeholder(tf.bool)

# data
normalized_imgs = normalize_images(images)
normalized_masks = normalize_images(masks)
cosinized_angles = deg2rad(angles)
relative_angles = cosinized_angles[:, -1, :] - cosinized_angles[:, 0, :]

base_imgs, target_imgs = split_imgs(normalized_imgs)
base_masks, target_masks = split_imgs(normalized_masks)
base_masks = 1 - base_masks
target_masks = 1 - target_masks

classes = classes[:, 0]

base_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
base_mask_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
target_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
target_mask_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 1]) # unused
angle_pl = tf.placeholder(tf.float32, [None, 2])
class_pl = tf.placeholder(tf.float32, None)


# model
concat_base = tf.concat([base_pl, base_mask_pl], axis=-1)
lv, ag_1, ag_2, ag_3 = model.encoder(concat_base, activation, is_training, batch_size)
merged_lv = model.merge_lv_angle(lv, angle_pl, activation)
gen_imgs = model.decoder(merged_lv, activation, is_training, ag_1, ag_2, ag_3)

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

saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/' + model_name + '_train')
    val_writer = tf.summary.FileWriter('summaries/' + model_name + '_val')
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    t_s, v_s = 0, 0
    for i in range(iters):
        base, bmasks, target, tmasks, angle = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles], feed_dict={handle: t_handle})
        resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
        resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)
        _, cost, summ = sess.run([train_op, mse_loss, loss_merged], feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                                               base_mask_pl: resized_bmasks, is_training: True})


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
            base, bmasks, target, tmasks, angle = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles], feed_dict={handle: t_handle})
            resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
            resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)
            img_summ = sess.run(img_merged, feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                       base_mask_pl: resized_bmasks, is_training: False})
            train_writer.add_summary(img_summ, t_s)
            train_writer.flush()
            t_s += 1

            base, bmasks, target, tmasks, angle = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles], feed_dict={handle: v_handle})
            resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
            resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)
            img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                                                                 base_mask_pl: resized_bmasks, is_training: False})
            print('VAL iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
            val_writer.add_summary(img_summ, v_s)
            val_writer.add_summary(loss_summ, v_s)
            val_writer.flush()
            v_s += 1