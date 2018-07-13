import datetime

import tensorflow as tf

from architectures.AE_only_conv import AE_only_conv
from dataproviders.chairs_provider import ChairProvider
from utils.functions import normalize_inputs

chairs_path = '../RotateNet_data/chairs'
batch_size = 50
learning_rate = 0.0005
n_examples = 1000
n_imgs = 2
img_size = 128
epochs = 30
ckpt = 20
activation = tf.nn.relu
note = 'mse_loss'

model = AE_only_conv()
model_name = 'chairs|--|{}|--|batch_{}|--|eta_{}|--|nexamples_{}|--|imgsize_{}|--|epochs_{}|--|activ_{}|--|nimgs_{}|--|date_{}|--|'.format(model.name, batch_size, learning_rate, n_examples,
                                                                                                                                       img_size, epochs, activation.__name__, n_imgs,
                                                                                                                                       datetime.datetime.now(), note)

provider = ChairProvider(chairs_path, n_examples=n_examples, n_imgs=n_imgs)

imgs = tf.placeholder(tf.float32, [None, None, img_size, img_size, 3])
angles = tf.placeholder(tf.float32, [None, None, 3])
is_training = tf.placeholder(tf.bool)

normalized_imgs, normalized_angles = normalize_inputs(imgs, angles)

base_imgs, target_imgs = model.split_imgs(normalized_imgs)
target_angles = normalized_angles[:, -1, :]

lv = model.encoder(base_imgs, activation, is_training)
merged_lv = model.merge_lv_angle(lv, target_angles, activation)
gen_imgs = model.decoder(merged_lv, activation, is_training)

mse_loss = tf.losses.mean_squared_error(labels=target_imgs, predictions=gen_imgs)

# summaries
concat_img = tf.concat([base_imgs, gen_imgs, target_imgs], 2)
loss_summary = tf.summary.scalar('loss', mse_loss)
img_summary = tf.summary.image('images', concat_img)
loss_merged = tf.summary.merge([loss_summary])
img_merged = tf.summary.merge([img_summary])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(mse_loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/' + model_name + '_train')
    val_writer = tf.summary.FileWriter('summaries/' + model_name + '_val')
    sess.run(tf.global_variables_initializer())

    num_batches = n_examples // batch_size
    for e in range(epochs):
        for i in range(num_batches):
            batch_imgs, batch_angles = provider.get_batch(batch_size, 'train', img_size)
            _, cost, summ = sess.run([train_op, mse_loss, loss_merged], feed_dict={imgs: batch_imgs, angles: batch_angles, is_training: True})
            print('Epoch {} of {}, train iter {} of {}, cost: {:.6f}'.format(e, epochs, i, num_batches, cost))
            train_writer.add_summary(summ, e * num_batches + i)
            train_writer.flush()

            if i % ckpt == 0:
                img_summ = sess.run(img_merged, feed_dict={imgs: batch_imgs, angles: batch_angles, is_training: False})
                train_writer.add_summary(img_summ, e * num_batches + i)
                train_writer.flush()

                batch_imgs, batch_angles = provider.get_batch(batch_size, 'val', img_size)
                img_summ, loss_summ, cost = sess.run([img_merged, loss_merged, mse_loss], feed_dict={imgs: batch_imgs, angles: batch_angles, is_training: False})
                print('Epoch {} of {}, val iter {} of {}, cost: {:.6f}'.format(e, epochs, i, num_batches, cost))
                val_writer.add_summary(img_summ, e * num_batches + i)
                val_writer.add_summary(loss_summ, e * num_batches + i)
                val_writer.flush()

                saver.save(sess, 'saved_models/' + model_name)
