import os

import tensorflow as tf

from training_module.architecture import build_model

build_model(True, 7)

#root = '/Users/Peace/Projects/halite3_ml/bots/duck_larger/'
#root = '/Users/Peace/Documents/models'
root = '/Users/Peace/Projects/c_5'

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(root, 'model_80000.ckpt'))
    
#    new_graph = tf.graph_util.convert_variables_to_constants(
#        sess,
#        tf.get_default_graph().as_graph_def(),
#        ['g_logits', 'm_logits'],
#        variable_names_whitelist=None,
#        variable_names_blacklist=None
#    )
#
#    with tf.gfile.GFile("/Users/Peace/Desktop/optimised_model.pb", "wb") as f:
#        f.write(new_graph.SerializeToString())

    saver.save(sess, '/Users/Peace/Desktop/halite_models/model.ckpt')



