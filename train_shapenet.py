from dataproviders.shapenet_provider import ShapenetProvider

data_path = '/media/carlo/My Files/Praca/Etap3/shapenet'
class_path = '/home/carlo/rotatenet/shapenet_raw'

a = ShapenetProvider(data_path, class_path, 30, 200, n_imgs=2)
handle, t_iter, v_iter, images, masks, angles, classes = a.dataset()
with tf.Session() as sess:
    t_handle, v_handle = sess.run([t_iter.string_handle(), v_iter.string_handle()])
    sess.run(tf.global_variables_initializer())

    for _ in range(1):
        i, m, a, c = sess.run([images, masks, angles, classes], feed_dict={handle: t_handle})
        print('angles: ', a)
        print('class: ', c)
        print(np.max(i), np.min(i))
        print(np.max(m), np.min(m))

        cv2.imshow('i1', i[0][0]/255)
        cv2.imshow('i2', i[0][1]/255)
        cv2.imshow('m1', m[0][0]/255)
        cv2.imshow('m2', m[0][1]/255)
        cv2.waitKey(-1)