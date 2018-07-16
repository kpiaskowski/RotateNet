import tensorflow as tf
import os
import numpy as np
import random


class ChairProvider:
    def __init__(self, path, batch_size, img_size, n_imgs=5, split_ratio=0.8, num_parallel_calls=12, seed=789432):
        self.batch_size = batch_size
        self.n_imgs = n_imgs
        self.img_size = img_size
        self.num_parallel_calls = num_parallel_calls
        self.class_paths = sorted([os.path.join(path, name) for name in os.listdir(path) if '.mat' not in name])

        ids = list(range(len(self.class_paths)))
        rng = random.Random(seed)
        rng.shuffle(ids)

        split_point = int(len(self.class_paths) * split_ratio)
        self.train_paths = self.class_paths[:split_point]
        self.val_paths = self.class_paths[split_point:]

    def load_images(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.img_size, self.img_size])
        return image_resized, filename

    def decode_string(self, img, filename):
        decoded = filename.decode()
        trans_dict = str.maketrans({'t': '', 'p': '', 'r': ''})
        split_str = decoded.split('/')

        angles = np.array(split_str[-1].rstrip('.png').translate(trans_dict).split('_')[2:], np.float32)
        class_path = '/'.join(split_str[:4])
        class_id = np.int32(self.class_paths.index(class_path))

        return img, angles, class_id

    def create_dataset(self, paths):
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths)))
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.flat_map(lambda dir: tf.data.Dataset.list_files(dir + '/renders/*.png').
                                   map(self.load_images, self.num_parallel_calls).
                                   map(lambda img, filename: tuple(tf.py_func(self.decode_string, [img, filename], [tf.float32, tf.float32, tf.int32], stateful=False)), self.num_parallel_calls).
                                   batch(self.n_imgs))
        dataset = dataset.filter(lambda x, y, z: tf.equal(tf.shape(y)[0], self.n_imgs))
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset

    def dataset(self):
        t_d = self.create_dataset(self.train_paths)
        v_d = self.create_dataset(self.val_paths)

        h = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(h, t_d.output_types, t_d.output_shapes)
        images, angles, classes = iter.get_next()

        t_iter = t_d.make_one_shot_iterator()
        v_iter = v_d.make_one_shot_iterator()

        return h, t_iter, v_iter, images, angles, classes
