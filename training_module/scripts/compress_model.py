import os

import tensorflow as tf

from training_module.architecture import build_model

build_model(True, 7)

root = '/Users/Peace/Projects/c_5'

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(root, 'model_72500.ckpt'))

    saver.save(sess, '/Users/Peace/Desktop/halite_models/model.ckpt')



