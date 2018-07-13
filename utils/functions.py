import tensorflow as tf

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
