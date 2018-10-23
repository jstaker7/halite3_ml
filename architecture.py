import tensorflow as tf

size = 32 # Single size for easier debugging (for now)
max_s = [1, 2, 2, 1]

input = tf.placeholder(tf.float32, [None, 256, 256, 5])

d_l1_a = tf.layers.conv2d(input, size, 3, activation=tf.nn.relu, padding='same')
d_l1_p = tf.nn.max_pool(d_l1_a, max_s, max_s, padding='VALID') # 128

d_l2_a = tf.layers.conv2d(d_l1_p, size, 3, activation=tf.nn.relu, padding='same')
d_l2_p = tf.nn.max_pool(d_l2_a, max_s, max_s, padding='VALID') # 64

d_l3_a = tf.layers.conv2d(d_l2_p, size, 3, activation=tf.nn.relu, padding='same')
d_l3_p = tf.nn.max_pool(d_l3_a, max_s, max_s, padding='VALID') # 32

d_l4_a = tf.layers.conv2d(d_l3_p, size, 3, activation=tf.nn.relu, padding='same')
d_l4_p = tf.nn.max_pool(d_l4_a, max_s, max_s, padding='VALID') # 16

d_l5_a = tf.layers.conv2d(d_l4_p, size, 3, activation=tf.nn.relu, padding='same')
d_l5_p = tf.nn.max_pool(d_l5_a, max_s, max_s, padding='VALID') # 8

d_l6_a = tf.layers.conv2d(d_l5_p, size, 3, activation=tf.nn.relu, padding='same')
d_l6_p = tf.nn.max_pool(d_l6_a, max_s, max_s, padding='VALID') # 4

d_l7_a = tf.layers.conv2d(d_l6_p, size, 3, activation=tf.nn.relu, padding='same')
d_l7_p = tf.nn.max_pool(d_l7_a, max_s, max_s, padding='VALID') # 2

d_l8_a = tf.layers.conv2d(d_l7_p, size, 3, activation=tf.nn.relu, padding='same')
d_l8_p = tf.nn.max_pool(d_l8_a, max_s, max_s, padding='VALID') # 1

latent = tf.layers.dense(d_l8_p, size, activation=tf.nn.relu)

u_l8_a = tf.layers.conv2d_transpose(latent, size, 3, 2, activation=tf.nn.relu, padding='same') # 2
u_l8_c = tf.concat([u_l8_a, d_l8_a], -1)
u_l8_s = tf.layers.conv2d(u_l8_c, size, 3, activation=tf.nn.relu, padding='same')

u_l7_a = tf.layers.conv2d_transpose(u_l8_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 4
u_l7_c = tf.concat([u_l7_a, d_l7_a], -1)
u_l7_s = tf.layers.conv2d(u_l7_c, size, 3, activation=tf.nn.relu, padding='same')

u_l6_a = tf.layers.conv2d_transpose(u_l7_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 8
u_l6_c = tf.concat([u_l6_a, d_l6_a], -1)
u_l6_s = tf.layers.conv2d(u_l6_c, size, 3, activation=tf.nn.relu, padding='same')

u_l5_a = tf.layers.conv2d_transpose(u_l6_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 16
u_l5_c = tf.concat([u_l5_a, d_l5_a], -1)
u_l5_s = tf.layers.conv2d(u_l5_c, size, 3, activation=tf.nn.relu, padding='same')

u_l4_a = tf.layers.conv2d_transpose(u_l5_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 32
u_l4_c = tf.concat([u_l4_a, d_l4_a], -1)
u_l4_s = tf.layers.conv2d(u_l4_c, size, 3, activation=tf.nn.relu, padding='same')

u_l3_a = tf.layers.conv2d_transpose(u_l4_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 64
u_l3_c = tf.concat([u_l3_a, d_l3_a], -1)
u_l3_s = tf.layers.conv2d(u_l3_c, size, 3, activation=tf.nn.relu, padding='same')

u_l2_a = tf.layers.conv2d_transpose(u_l3_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 128
u_l2_c = tf.concat([u_l2_a, d_l2_a], -1)
u_l2_s = tf.layers.conv2d(u_l2_c, size, 3, activation=tf.nn.relu, padding='same')

u_l1_a = tf.layers.conv2d_transpose(u_l2_s, size, 3, 2, activation=tf.nn.relu, padding='same') # 256
u_l1_c = tf.concat([u_l1_a, d_l1_a], -1)
u_l1_s = tf.layers.conv2d(u_l1_c, size, 3, activation=tf.nn.relu, padding='same')

print(u_l1_s.get_shape())

