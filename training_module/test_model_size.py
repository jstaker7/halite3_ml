import os

import tensorflow as tf

from training_module.architecture import build_model

build_model(True)

with tf.Session() as sess:
    saver = tf.train.Saver()
    tf.initializers.global_variables().run()
    saver.save(sess, '/Users/Peace/Projects/halite3_ml/scratch/test.ckpt')




