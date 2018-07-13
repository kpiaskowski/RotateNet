import os
import random
import time

import cv2
import numpy as np


class ChairProvider:
    def __init__(self, path, n_examples, n_imgs=5, split_ratio=0.8, seed=789432):
        """
        :param path: path to rendered chairs directory
        :param img_size: desired size of images, they will be resized to match this size
        :param n_imgs: number of images in single training example
        :param batch_size: size of batch
        :param n_examples: number of predefined examples in the whole dataset
        :param split_ratio: split train/val set according to this ratio
        """
        self.n_examples = n_examples
        self.path = path

        self.names_angles = self.read_filenames(self.path)

        print('Generating data...')
        s = time.time()
        self.train, self.val = self.generate_examples(n_examples, self.names_angles, n_imgs, split_ratio, seed)
        e = time.time()
        print("Data generated in {:.2f} seconds".format(e - s))

    def get_batch(self, batch_size, type, img_size):
        """
        Reads single batch from disc.
        :param batch_size: number of examples in batch (note that batch = batchsize * n_imgs_per_batch)
        :param type: 'val' or 'train'
        :param img_size: size of output image
        :return:
        """
        imnames, angles = self.train if type == 'train' else self.val
        datalen = len(imnames)

        ids = [random.randint(0, datalen - 1) for _ in range(batch_size)]
        batch_angles = [angles[i] for i in ids]
        batches_names = [imnames[i] for i in ids]
        imgs = [[cv2.resize(np.array(cv2.imread(name), np.float32), (img_size, img_size)) for name in batch_name] for batch_name in batches_names]
        return imgs, batch_angles

    def read_filenames(self, dir):
        """
        Reads all names from the root directory of chairs dataset. Groups them by their class.
        :param dir: root chairs directory
        """
        class_names = sorted((name for name in os.listdir(dir) if 'all_chair_names.mat' not in name))

        data = []
        for i, class_name in enumerate(class_names):
            class_path = os.path.join(dir, class_name, 'renders')
            filepaths = sorted((os.path.join(class_path, filename) for filename in os.listdir(class_path)))
            angles = [self.read_angle_from_name(path) for path in filepaths]
            data.append((filepaths, angles))

        return data

    def read_angle_from_name(self, name):
        """
        Reads the vector of angles from filename
        :param name: string with filename, could be either full path or relative path
        Returns: list of 3 angles
        """
        trans_dict = str.maketrans({'t': '', 'p': '', 'r': ''})
        angles = np.array(name.split('/')[-1].rstrip('.png').translate(trans_dict).split('_')[2:], np.float)
        return angles

    def generate_examples(self, n_examples, data, n_imgs, split_ratio, seed):
        """
        Generates examples randomly by grouping their names into single list. Pickles generated data to speed up the future queries.
        :param seed: used in random generators
        :param n_imgs: number of images and angles per examples
        :param filenames: dictionary with data
        :return:
        """
        rng = random.Random(seed)
        img_paths = []
        relative_angles = []

        datalen = len(data)
        for i in range(n_examples):
            cls = rng.randint(0, datalen - 1)
            class_data = data[cls]
            names, angles = class_data[0], class_data[1]

            class_len = len(names)
            ids = [rng.randint(0, class_len - 1) for _ in range(n_imgs)]

            paths = [names[i] for i in ids]
            base_angle = angles[ids[0]]
            angles = [angles[i] - base_angle for i in ids]

            img_paths.append(paths)
            relative_angles.append(angles)

        # train / val
        split_point = int(n_examples * split_ratio)
        return (img_paths[:split_point], relative_angles[:split_point]), (img_paths[split_point:], relative_angles[split_point:])
