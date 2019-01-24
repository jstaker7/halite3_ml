import os
import copy

import numpy as np
import tensorflow as tf

cd = '/Users/Peace/Projects/halite3_ml/bots/duck_v31'

with tf.Session() as sess:
    tf.train.import_meta_graph(os.path.join(cd, "model.ckpt.meta"))
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(cd, "model.ckpt"))
    vars = [ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                    if 'bias' not in v.name and 'c' == v.name[0]]
    L2_loss = tf.add_n(vars) * 0.00005
                    
    print(len(vars))

    var = [v for v in tf.trainable_variables()]
    
    var = sess.run(var)

    print(np.min([np.min(x) for x in var]))
    print(np.mean([np.mean(x) for x in var]))
    print(np.max([np.max(x) for x in var]))

