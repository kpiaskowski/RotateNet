import cv2

import numpy as np

import tensorflow as tf

from architectures.AE_only_conv import AE_only_conv
from dataproviders.reduced_shapenet_provider import ShapenetProvider
from utils.functions import cropping_pipeline, normalize_images, split_imgs, deg2rad, find_bbox_coords, crop_imgs, resize_and_pad

batch_size = 1
learning_rate = 0.0005
n_imgs = 2
img_size = 128
iters = 500000
ckpt = 20
mean_img_val = 0.75
save_ckpt = 10000
activation = tf.nn.relu
note = 'cropped with masks'
model_path = '/home/carlo/rotatenet/RotateNet/saved_models/continued chair rel angles 6 step---date_2018-09-05 09:17:26.488222/model.ckpt-20000'

# dataprovider
dataprovider = ShapenetProvider('../shapenet', '../shapenet_raw', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
handle, t_iter, v_iter, images, masks, depths, angles, classes = dataprovider.dataset()

# namingyou
model = AE_only_conv()

is_training = tf.placeholder(tf.bool)

# data
normalized_imgs = normalize_images(images)
normalized_masks = normalize_images(masks)
normalized_depths = normalize_images(depths)

cosinized_angles = deg2rad(angles)
relative_angles = cosinized_angles[:, -1, :]#  - cosinized_angles[:, 0, :]

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

concat_base = tf.concat([base_pl, base_mask_pl], axis=-1)
lv, ag_1, ag_2, ag_3 = model.encoder(concat_base, activation, is_training, batch_size)
merged_lv = model.merge_lv_angle(lv, angle_pl, activation)
gen_imgs, gen_depths = model.decoder(merged_lv, activation, is_training, ag_1, ag_2, ag_3)

saver = tf.train.Saver()
with tf.Session() as sess:
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    base, bmasks, target, tmasks, angle, tdepth = sess.run([base_imgs, base_masks, target_imgs, target_masks, relative_angles, target_depths], feed_dict={handle: v_handle})
    resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
    resized_target, resized_tmasks = cropping_pipeline(target, tmasks, img_size, mean_img_val)
    xmin, xmax, ymin, ymax, *_ = find_bbox_coords(tmasks)
    cropped_depths = crop_imgs(tdepth, xmin, xmax, ymin, ymax)
    resized_depths = resize_and_pad(cropped_depths, img_size, mean_img_val, 1)
    generated_rgb, generated_depth = sess.run([gen_imgs, gen_depths], feed_dict={base_pl: resized_base, angle_pl: angle,
                                                                                 base_mask_pl: resized_bmasks, is_training: False})
    for a1 in range(0, 31, 10):
        for a2 in range(0, 361, 12):
            print(a1, a2)
            angles_delta = [a1, a2]
            cos_angles = deg2rad(np.expand_dims(np.array(angles_delta, np.float32), 0))

            generated_rgb, generated_depth = sess.run([gen_imgs, gen_depths], feed_dict={base_pl: resized_base, angle_pl: cos_angles,
                                                                                         base_mask_pl: resized_bmasks, is_training: False})

            base_img = np.uint8(resized_base[0] / np.max(resized_base[0]) * 180)
            target_rgb = np.uint8(generated_rgb[0] / np.max(generated_depth[0]) * 255)
            target_depth = np.uint8(generated_depth[0] / np.max(generated_depth[0]) * 255)

            cv2.imshow('base', base_img)
            cv2.imshow('rgb', target_rgb)
            cv2.imshow('depth', target_depth)

            cv2.imwrite('samples/sample1/rgb_{}_{}.png'.format(a1, a2), target_rgb)
            cv2.imwrite('samples/sample1/depth_{}_{}.png'.format(a1, a2), target_depth)
            cv2.waitKey(200)
    cv2.imwrite('samples/sample1/base.png', np.uint8(resized_base[0] / np.max(resized_base[0]) * 255))
