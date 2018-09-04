# from architectures.AE_only_conv import AE_only_conv
# from dataproviders.chairs_provider import ChairProvider
# import tensorflow as tf
# import datetime
# import numpy as np
# from utils.functions import cropping_pipeline, normalize_images, cosinize_angles_np, split_imgs
# import cv2
#
# batch_size = 1
# learning_rate = 0.0005
# n_imgs = 2
# img_size = 128
# iters = 500000
# ckpt = 20
# mean_img_val = 0.75
# save_ckpt = 10000
# activation = tf.nn.relu
# note = 'cropped with masks'
# model_path = '/home/carlo/rotatenet/RotateNet/saved_models/reduced shapene bottle---date_2018-08-29 11:15:23.394629/model.ckpt-70000'
#
# # dataprovider
# dataprovider = ChairProvider('../RotateNet_data/chairs', batch_size=batch_size, img_size=img_size, n_imgs=n_imgs)
# handle, t_iter, v_iter, images, angles, classes = dataprovider.dataset()
#
# # namingyou
# model = AE_only_conv()
#
# # placeholders
# is_training = tf.placeholder(tf.bool)
#
# # data
# normalized_imgs = normalize_images(images)
# base_imgs, _ = split_imgs(normalized_imgs)
# base_masks = 1 - tf.to_float(tf.equal(base_imgs, 1))[..., :1]  # preserve dims
#
# base_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
# base_mask_pl = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
# angle_pl = tf.placeholder(tf.float32, [None, 3])
#
# # model
# concat_base = tf.concat([base_pl, base_mask_pl], axis=-1)
# lv = model.encoder(concat_base, activation, is_training, 1)
# merged_lv = model.merge_lv_angle(lv, angle_pl, activation)
# gen_imgs = model.decoder(merged_lv, activation, is_training, 1)
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, model_path)
#
#     base, bmasks = sess.run([base_imgs, base_masks], feed_dict={handle: v_handle})
#
#     resized_base, resized_bmasks = cropping_pipeline(base, bmasks, img_size, mean_img_val)
#
#     raw_imgs = []
#     for a1 in range(10, 21, 10):
#         for a2 in range(0, 361, 5):
#             angles_delta = [a1, a2, 96]
#             cos_angles = cosinize_angles_np(np.expand_dims(np.array(angles_delta, np.float32), 0))
#
#             outputs = sess.run(gen_imgs, feed_dict={base_pl: resized_base,
#                                                     base_mask_pl: resized_bmasks,
#                                                     angle_pl: cos_angles,
#                                                     is_training: False})
#             concat_in_out = np.concatenate([resized_base[0], outputs[0]], 1)
#             raw_imgs.append(concat_in_out)
#             # print(angles_delta)
#             # cv2.imshow('gen', concat_in_out)
#             # cv2.waitKey(1000)
#     ls = [raw_imgs[i] for i in range(0, len(raw_imgs), 2)]
#     rs = [raw_imgs[i] for i in range(1, len(raw_imgs), 2)]
#     togetger = ls + rs
#
#     for img in togetger:
#         cv2.imshow('gen', img)
#         cv2.waitKey(200)
#
