from architectures.AE_only_conv import AE_only_conv
from dataproviders.reduced_shapenet_provider import ShapenetProvider
import tensorflow as tf
import datetime

from utils.functions import cropping_pipeline, normalize_images, split_imgs, deg2rad, find_bbox_coords, crop_imgs, resize_and_pad
import os

batch_size = 30
learning_rate = 0.0005
n_imgs = 2
img_size = 128
iters = 500000
ckpt = 20
save_ckpt = 10000
activation = tf.nn.relu
mean_img_val = 0.75
note = 'reduced shapenet'

# dataproviderp
dataprovider = ShapenetProvider('../shapenet', '../shapenet_raw', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
handle, t_iter, v_iter, images, masks, depths, angles, classes = dataprovider.dataset()

# namingyou
model = AE_only_conv()
model_name = '{}---date_{}'.format(note, datetime.datetime.now())

# placeholders
is_training = tf.placeholder(tf.bool)

# data
normalized_imgs = normalize_images(images)
normalized_masks = normalize_images(masks)
normalized_depths = normalize_images(depths)

cosinized_angles = deg2rad(angles)
relative_angles = cosinized_angles[:, -1, :]# - cosinized_angles[:, 0, :]

base_imgs, target_imgs = split_imgs(normalized_imgs)
base_masks, target_masks = split_imgs(normalized_masks)
_, target_depths = split_imgs(normalized_depths)

base_masks = 1 - base_masks
target_masks = 1 - target_masks

classes = classes[:, 0]

base_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
base_mask_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
target_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
target_depth_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
angle_pl = tf.placeholder(tf.float32, [None, 2])
class_pl = tf.placeholder(tf.float32, None)

# model
concat_base = tf.concat([base_pl, base_mask_pl], axis=-1)
lv, ag_1, ag_2, ag_3 = model.encoder(concat_base, activation, is_training, batch_size)
merged_lv = model.merge_lv_angle(lv, angle_pl, activation)
gen_imgs, gen_depths = model.decoder(merged_lv, activation, is_training, ag_1, ag_2, ag_3)

# losses
mse_loss = tf.losses.mean_squared_error(labels=target_pl, predictions=gen_imgs)
depth_loss = tf.losses.mean_squared_error(labels=target_depth_pl, predictions=gen_depths)
total_loss = mse_loss + depth_loss

# summaries
concat_img = tf.concat([base_pl, gen_imgs, target_pl, tf.image.grayscale_to_rgb(gen_depths), tf.image.grayscale_to_rgb(target_depth_pl)], 2)
rgb_loss_summary = tf.summary.scalar('rgb_loss', mse_loss)
depth_loss_summary = tf.summary.scalar('depth_loss', depth_loss)
total_loss_summary = tf.summary.scalar('total_loss', total_loss)

img_summary = tf.summary.image('images', concat_img)

loss_merged = tf.summary.merge([rgb_loss_summary, depth_loss_summary, total_loss_summary])
img_merged = tf.summary.merge([img_summary])

# train ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(total_loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
    train_op = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/' + model_name + '_train')
    val_writer = tf.summary.FileWriter('summaries/' + model_name + '_val')
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    t_s, v_s = 0, 0
    for i in range(iters):
        try:
            base, bmasks, target, tmasks, angle, tdepth = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles, target_depths], feed_dict={handle: t_handle})
            resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
            resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)

            xmin, xmax, ymin, ymax, *_ = find_bbox_coords(tmasks)
            cropped_depths = crop_imgs(tdepth, xmin, xmax, ymin, ymax)
            resized_depths = resize_and_pad(cropped_depths, img_size, mean_img_val, 1)

            _, cost, summ = sess.run([train_op, total_loss, loss_merged], feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                                                     base_mask_pl: resized_bmasks, is_training: True,
                                                                                     target_depth_pl: resized_depths})

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
                base, bmasks, target, tmasks, angle, tdepth = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles, target_depths], feed_dict={handle: t_handle})
                resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
                resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)

                xmin, xmax, ymin, ymax, *_ = find_bbox_coords(tmasks)
                cropped_depths = crop_imgs(tdepth, xmin, xmax, ymin, ymax)
                resized_depths = resize_and_pad(cropped_depths, img_size, mean_img_val, 1)

                img_summ = sess.run(img_merged, feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                           base_mask_pl: resized_bmasks, is_training: False,
                                                           target_depth_pl: resized_depths})
                train_writer.add_summary(img_summ, t_s)
                train_writer.flush()
                t_s += 1

                base, bmasks, target, tmasks, angle, tdepth = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles, target_depths], feed_dict={handle: v_handle})
                resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
                resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)

                xmin, xmax, ymin, ymax, *_ = find_bbox_coords(tmasks)
                cropped_depths = crop_imgs(tdepth, xmin, xmax, ymin, ymax)
                resized_depths = resize_and_pad(cropped_depths, img_size, mean_img_val, 1)

                img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={base_pl: resized_base, target_pl: resized_target, angle_pl: angle,
                                                                                                     base_mask_pl: resized_bmasks, is_training: False,
                                                                                                     target_depth_pl: resized_depths})
                print('VAL iteration {} of {}, cost: {:.6f}'.format(i, iters, cost))
                val_writer.add_summary(img_summ, v_s)
                val_writer.add_summary(loss_summ, v_s)
                val_writer.flush()
                v_s += 1

        except:
            pass
