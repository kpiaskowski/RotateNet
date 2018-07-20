import tensorflow as tf
import cv2
import numpy as np

pi_on_180 = 0.017453292519943295


def rad2deg(rad):
    return rad / pi_on_180


def deg2rad(deg):
    return deg * pi_on_180


def normalize_inputs(imgs, angles):
    """Normalizes images and converts angles to rads and cosines"""
    norm_imgs = imgs / 255
    rad_ang = deg2rad(angles)
    cos_ang = tf.cos(rad_ang)
    return norm_imgs, cos_ang

def mask_images(images, masks):
    return images * masks

def find_bbox_coords(masks):
    xmin, xmax, ymin, ymax, h, w = [],[],[],[],[],[]
    for mask in masks:
        points = cv2.findNonZero(np.uint8(mask))
        bbox = cv2.boundingRect(points)
        xmin.append(bbox[0])
        ymin.append(bbox[1])
        xmax.append(bbox[0]+bbox[2])
        ymax.append(bbox[1]+bbox[3])
        w.append(bbox[2])
        h.append(bbox[3])

    return xmin, xmax, ymin, ymax, h, w


def crop_imgs(images, xmin, xmax, ymin, ymax):
    bsize = images.shape[0]
    cropped_imgs = []
    for i in range(bsize):
        cropped_imgs.append(images[i][ymin[i]:ymax[i], xmin[i]:xmax[i]])
    return cropped_imgs


def resize_and_pad(images, img_size):
    resized_imgs = []
    for img in images:
        y, x = img.shape[0], img.shape[1]
        if y >= x:
            new_x = int(img_size / y * x / 2) * 2  # rounding to nearest
            resized = cv2.resize(img, (new_x, img_size))
            padded = np.pad(resized, ((0,0), ((img_size - new_x)//2, (img_size - new_x)//2), (0, 0)), mode='constant', constant_values=1)
            resized_imgs.append(padded)
        else:
            new_y = int(img_size / x * y / 2) * 2  # rounding to nearest)
            resized = cv2.resize(img, (img_size, new_y))
            padded = np.pad(resized, (((img_size - new_y)//2, (img_size - new_y)//2), (0,0), (0, 0)), mode='constant', constant_values=1)
            resized_imgs.append(padded)
    return resized_imgs

def cropping_pipeline(imgs, masks, img_size):
    xmin, xmax, ymin, ymax, *_ = find_bbox_coords(masks)
    cropped = crop_imgs(imgs, xmin, xmax, ymin, ymax)
    resized = resize_and_pad(cropped, img_size)
    return resized