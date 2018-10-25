import tensorflow as tf

def build_model():

    size = 8 # Single size for easier debugging (for now)
    max_s = [1, 2, 2, 1]
    learning_rate = 0.0005

    frames = tf.placeholder(tf.float32, [None, 256, 256, 5])
    can_afford = tf.placeholder(tf.float32, [None, 3])
    turns_left = tf.placeholder(tf.float32, [None, 1])
    my_ships = tf.placeholder(tf.uint8, [None, 256, 256, 1])
    
    my_ships = tf.cast(my_ships, tf.float32)

    moves = tf.placeholder(tf.uint8, [None, 256, 256, 1])
    generate = tf.placeholder(tf.float32, [None, 1])

    tf.add_to_collection('frames', frames)
    tf.add_to_collection('can_afford', can_afford)
    tf.add_to_collection('turns_left', turns_left)
    tf.add_to_collection('my_ships', my_ships)
    tf.add_to_collection('moves', moves)
    tf.add_to_collection('generate', generate)

    moves = tf.one_hot(moves, 6)

    ca = tf.layers.dense(can_afford, 16)
    tl = tf.layers.dense(turns_left, 16)

    ca = tf.expand_dims(ca, 1)
    ca = tf.expand_dims(ca, 1)
    tl = tf.expand_dims(tl, 1)
    tl = tf.expand_dims(tl, 1)

    d_l1_a = tf.layers.conv2d(frames, 16, 3, activation=tf.nn.relu, padding='same')
    d_l1_p = tf.nn.max_pool(d_l1_a, max_s, max_s, padding='VALID') # 128

    d_l2_a = tf.layers.conv2d(d_l1_p, 32, 3, activation=tf.nn.relu, padding='same')
    d_l2_p = tf.nn.max_pool(d_l2_a, max_s, max_s, padding='VALID') # 64

    d_l3_a = tf.layers.conv2d(d_l2_p, 32, 3, activation=tf.nn.relu, padding='same')
    d_l3_p = tf.nn.max_pool(d_l3_a, max_s, max_s, padding='VALID') # 32

    d_l4_a = tf.layers.conv2d(d_l3_p, 64, 3, activation=tf.nn.relu, padding='same')
    d_l4_p = tf.nn.max_pool(d_l4_a, max_s, max_s, padding='VALID') # 16

    d_l5_a = tf.layers.conv2d(d_l4_p, 64, 3, activation=tf.nn.relu, padding='same')
    d_l5_p = tf.nn.max_pool(d_l5_a, max_s, max_s, padding='VALID') # 8

    d_l6_a = tf.layers.conv2d(d_l5_p, 128, 3, activation=tf.nn.relu, padding='same')
    d_l6_p = tf.nn.max_pool(d_l6_a, max_s, max_s, padding='VALID') # 4

    d_l7_a = tf.layers.conv2d(d_l6_p, 128, 3, activation=tf.nn.relu, padding='same')
    d_l7_p = tf.nn.max_pool(d_l7_a, max_s, max_s, padding='VALID') # 2

    d_l8_a = tf.layers.conv2d(d_l7_p, 256, 3, activation=tf.nn.relu, padding='same')
    d_l8_p = tf.nn.max_pool(d_l8_a, max_s, max_s, padding='VALID') # 1

    final_state = tf.concat([d_l8_p, ca, tl], -1)
    latent = tf.layers.dense(final_state, 256, activation=tf.nn.relu)

    u_l8_a = tf.layers.conv2d_transpose(latent, 1024, 3, 2, activation=tf.nn.relu, padding='same') # 2
    u_l8_c = tf.concat([u_l8_a, d_l8_a], -1)
    u_l8_s = tf.layers.conv2d(u_l8_c, 1024, 3, activation=tf.nn.relu, padding='same')

    u_l7_a = tf.layers.conv2d_transpose(u_l8_s, 512, 3, 2, activation=tf.nn.relu, padding='same') # 4
    u_l7_c = tf.concat([u_l7_a, d_l7_a], -1)
    u_l7_s = tf.layers.conv2d(u_l7_c, 512, 3, activation=tf.nn.relu, padding='same')

    u_l6_a = tf.layers.conv2d_transpose(u_l7_s, 256, 3, 2, activation=tf.nn.relu, padding='same') # 8
    u_l6_c = tf.concat([u_l6_a, d_l6_a], -1)
    u_l6_s = tf.layers.conv2d(u_l6_c, 256, 3, activation=tf.nn.relu, padding='same')

    u_l5_a = tf.layers.conv2d_transpose(u_l6_s, 128, 3, 2, activation=tf.nn.relu, padding='same') # 16
    u_l5_c = tf.concat([u_l5_a, d_l5_a], -1)
    u_l5_s = tf.layers.conv2d(u_l5_c, 128, 3, activation=tf.nn.relu, padding='same')

    u_l4_a = tf.layers.conv2d_transpose(u_l5_s, 64, 3, 2, activation=tf.nn.relu, padding='same') # 32
    u_l4_c = tf.concat([u_l4_a, d_l4_a], -1)
    u_l4_s = tf.layers.conv2d(u_l4_c, 64, 3, activation=tf.nn.relu, padding='same')

    u_l3_a = tf.layers.conv2d_transpose(u_l4_s, 32, 3, 2, activation=tf.nn.relu, padding='same') # 64
    u_l3_c = tf.concat([u_l3_a, d_l3_a], -1)
    u_l3_s = tf.layers.conv2d(u_l3_c, 32, 3, activation=tf.nn.relu, padding='same')

    u_l2_a = tf.layers.conv2d_transpose(u_l3_s, 32, 3, 2, activation=tf.nn.relu, padding='same') # 128
    u_l2_c = tf.concat([u_l2_a, d_l2_a], -1)
    u_l2_s = tf.layers.conv2d(u_l2_c, 32, 3, activation=tf.nn.relu, padding='same')

    u_l1_a = tf.layers.conv2d_transpose(u_l2_s, 64, 3, 2, activation=tf.nn.relu, padding='same') # 256
    u_l1_c = tf.concat([u_l1_a, d_l1_a], -1)
    u_l1_s = tf.layers.conv2d(u_l1_c, 63, 3, activation=tf.nn.relu, padding='same')

    moves_logits = tf.layers.conv2d(u_l1_s, 6, 3, activation=None, padding='same')

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=moves,
                                                logits=moves_logits,
                                                dim=-1)

    losses = tf.expand_dims(losses, -1)

    masked_loss = losses * my_ships

    ships_per_frame = tf.reduce_sum(my_ships, axis=[1, 2])

    frame_loss = tf.reduce_sum(masked_loss, axis=[1, 2])

    average_frame_loss = frame_loss / (ships_per_frame + 0.00000001) # First frames have no ship

    loss = tf.reduce_mean(average_frame_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)

    return


