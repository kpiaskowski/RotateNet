import os
import random

import numpy as np
import tensorflow as tf


class ShapenetProvider:
    def __init__(self, data_path, class_path, batch_size, img_size, n_imgs=5, split_ratio=0.8, num_parallel_calls=12, seed=789432):
        self.batch_size = batch_size
        self.n_imgs = n_imgs
        self.img_size = img_size
        self.num_parallel_calls = num_parallel_calls

        # self.classes = sorted(os.listdir(class_path)) temporarily  only armchairs
        self.classes = ['armchair']
        self.data_dirs = sorted([os.path.join(data_path, name) for name in os.listdir(data_path) if '.mat' not in name])

        self.train_paths, self.val_paths = self.split_sets(split_ratio, self.classes, self.data_dirs, seed)

    def filter_name(self, name):
        """../../shapenet/armchair_213/20_99_rgb.png"""
        ang1, ang2, _ = name.decode().split('/')[-1].split('_')
        ang1, ang2 = int(ang1), int(ang2)
        if (ang1 == 10 or ang1 == 20) and ang2 % 12 == 0:
            return True
        else:
            return False

    def create_dataset(self, paths):
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths)))
        dataset = dataset.shuffle(20000)
        dataset = dataset.flat_map(lambda dir: tf.data.Dataset.list_files(dir + '/*rgb.png').
                                   shuffle(30).
                                   filter(lambda name: tf.py_func(self.filter_name, [name], tf.bool, stateful=False)).
                                   map(self.load_images, self.num_parallel_calls).
                                   map(
            lambda img, mask, depth, filename: tuple(tf.py_func(self.decode_name, [img, mask, depth, filename], [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32], stateful=False)),
            self.num_parallel_calls).
                                   batch(self.n_imgs))
        dataset = dataset.filter(lambda x, y, z, v, w: tf.equal(tf.shape(y)[0], self.n_imgs))
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.shuffle(150)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat()
        return dataset

    def dataset(self):
        t_d = self.create_dataset(self.train_paths)
        v_d = self.create_dataset(self.val_paths)

        h = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(h, t_d.output_types, t_d.output_shapes)
        images, masks, depths, angles, classes = iter.get_next()

        t_iter = t_d.make_one_shot_iterator()
        v_iter = v_d.make_one_shot_iterator()

        return h, t_iter, v_iter, images, masks, depths, angles, classes

    def load_images(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.img_size, self.img_size])

        mask_string = tf.read_file(tf.regex_replace(filename, "rgb", "mask"))
        mask_decoded = tf.image.decode_png(mask_string)
        mask_resized = tf.image.resize_images(mask_decoded, [self.img_size, self.img_size])
        mask_gray = tf.image.rgb_to_grayscale(mask_resized)

        depth_string = tf.regex_replace(filename, "rgb", "depth")
        depth_string = tf.regex_replace(depth_string, "shapenet", "shapenet_depth")
        depth_string = tf.read_file(depth_string)
        depth_decoded = tf.image.decode_png(depth_string)
        depth_resized = tf.image.resize_images(depth_decoded, [self.img_size, self.img_size])
        depth_gray = tf.image.rgb_to_grayscale(depth_resized)

        return image_resized, mask_gray, depth_gray, filename

    def decode_name(self, img, mask, depth, filename):
        decoded = filename.decode()
        split_str = ''.join(decoded.split('/')[-2].split('_')[:-1])
        classes = np.int32(self.classes.index(split_str))
        angles = np.float32(decoded.split('/')[-1].split('_')[:2])
        return img, mask, depth, angles, classes

    def split_sets(self, ratio, classes, data_dirs, seed):
        rng = random.Random(seed)
        train_paths = []
        val_paths = []
        for class_ in classes:
            dirs = sorted([dir_ for dir_ in data_dirs if class_ in dir_])
            if dirs:
                split_point = int(ratio * len(dirs))
                rng.shuffle(dirs)
                train_paths.extend(dirs[:split_point])
                val_paths.extend(dirs[split_point:])
        return train_paths, val_paths
