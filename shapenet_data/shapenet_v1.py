from distutils.dir_util import copy_tree
import os
from math import radians as rad
import json
from itertools import chain
from shutil import copyfile

root_dir = '/home/carlo/rotatenet/shapenet_raw'
dst_dir = '/media/carlo/My Files/Praca/Etap3/shapenet'

classes = list(sorted(os.listdir(root_dir)))

for cls in classes:
    obj_dirs = sorted([os.path.join(root_dir, cls, name) for name in os.listdir(os.path.join(root_dir, cls))])
    for i, obj_dir in enumerate(obj_dirs):
        # creating directories
        # os.mkdir(os.path.join(dst_dir, cls + '_{}'.format(i)))

        # blender goes here

        print('Class {}, file {} of {}'.format(cls, i + 1, len(obj_dirs)))
        print(os.listdir(obj_dir))

lamp_xy_range = 7
lamp_z_range = 5

positions = [[-lamp_xy_range, -lamp_xy_range, -lamp_z_range], [lamp_xy_range, -lamp_xy_range, -lamp_z_range], [-lamp_xy_range, lamp_xy_range, -lamp_z_range], [lamp_xy_range, lamp_xy_range, -lamp_z_range],
 [-lamp_xy_range, -lamp_xy_range, lamp_z_range], [lamp_xy_range, -lamp_xy_range, lamp_z_range], [-lamp_xy_range, lamp_xy_range, lamp_z_range], [lamp_xy_range, lamp_xy_range, lamp_z_range]]
