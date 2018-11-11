import tensorflow as tf

def build_model(inference=False):

    learning_rate = 0.0006

    frames = tf.placeholder(tf.float32, [None, 128, 128, 5])
    can_afford = tf.placeholder(tf.float32, [None, 3])
    turns_left = tf.placeholder(tf.float32, [None, 1])
    my_ships = tf.placeholder(tf.uint8, [None, 128, 128, 1])
    
    my_ships = tf.cast(my_ships, tf.float32)

    moves = tf.placeholder(tf.uint8, [None, 128, 128, 1])
    generate = tf.placeholder(tf.float32, [None, 1])
    is_training = tf.placeholder(tf.bool)

    tf.add_to_collection('frames', frames)
    tf.add_to_collection('can_afford', can_afford)
    tf.add_to_collection('turns_left', turns_left)
    tf.add_to_collection('my_ships', my_ships)
    tf.add_to_collection('moves', moves)
    tf.add_to_collection('generate', generate)
    tf.add_to_collection('is_training', is_training)

    moves = tf.one_hot(moves, 6)

    ca = tf.layers.dense(can_afford, 16)
    tl = tf.layers.dense(turns_left, 16)

    ca = tf.expand_dims(ca, 1)
    ca = tf.expand_dims(ca, 1)
    tl = tf.expand_dims(tl, 1)
    tl = tf.expand_dims(tl, 1)

#    d_l1_a = tf.layers.conv2d(frames, 16, 3, activation=tf.nn.relu, padding='same')
#    d_l1_p = tf.nn.max_pool(d_l1_a, max_s, max_s, padding='VALID') # 128

    d_l2_a_1 = tf.layers.conv2d(frames, 32, 5, activation=tf.nn.relu, padding='same')
    d_l2_a_1 = tf.layers.batch_normalization(d_l2_a_1, training=is_training, name='bn1')
    d_l2_a_2 = tf.layers.conv2d(d_l2_a_1, 32, 5, activation=tf.nn.relu, padding='same')
    d_l2_a_2 = tf.layers.batch_normalization(d_l2_a_2, training=is_training, name='bn2')
    d_l2_a_3 = tf.layers.conv2d(d_l2_a_2, 32, 5, activation=tf.nn.relu, padding='same')
    d_l2_a_3 = tf.layers.batch_normalization(d_l2_a_3, training=is_training, name='bn3')
    d_l2_p = tf.layers.conv2d(d_l2_a_3, 32, 5, stride=2, activation=tf.nn.relu, padding='valid') # 64
    d_l2_p = tf.layers.batch_normalization(d_l2_p, training=is_training, name='bn4')

    d_l3_a = tf.layers.conv2d(d_l2_p, 32, 3, activation=tf.nn.relu, padding='same')
    d_l3_a = tf.layers.batch_normalization(d_l3_a, training=is_training, name='bn5')
    d_l3_p = tf.layers.conv2d(d_l3_a, 32, 3, stride=2, activation=tf.nn.relu, padding='valid') # 32
    d_l3_p = tf.layers.batch_normalization(d_l3_p, training=is_training, name='bn6')

    d_l4_a = tf.layers.conv2d(d_l3_p, 64, 3, activation=tf.nn.relu, padding='same')
    d_l4_a = tf.layers.batch_normalization(d_l4_a, training=is_training, name='bn7')
    d_l4_p = tf.layers.conv2d(d_l4_a, 64, 3, stride=2, activation=tf.nn.relu, padding='valid') # 16
    d_l4_p = tf.layers.batch_normalization(d_l4_p, training=is_training, name='bn8')

    d_l5_a = tf.layers.conv2d(d_l4_p, 64, 3, activation=tf.nn.relu, padding='same')
    d_l5_a = tf.layers.batch_normalization(d_l5_a, training=is_training, name='bn9')
    d_l5_p = tf.layers.conv2d(d_l5_a, 64, 3, stride=2, activation=tf.nn.relu, padding='valid') # 8
    d_l5_p = tf.layers.batch_normalization(d_l5_p, training=is_training, name='bn10')

    d_l6_a = tf.layers.conv2d(d_l5_p, 128, 3, activation=tf.nn.relu, padding='same')
    d_l6_a = tf.layers.batch_normalization(d_l6_a, training=is_training, name='bn11')
    d_l6_p = tf.layers.conv2d(d_l6_a, 128, 3, stride=2, activation=tf.nn.relu, padding='valid') # 4
    d_l6_p = tf.layers.batch_normalization(d_l6_p, training=is_training, name='bn12')

    d_l7_a = tf.layers.conv2d(d_l6_p, 128, 3, activation=tf.nn.relu, padding='same')
    d_l7_a = tf.layers.batch_normalization(d_l7_a, training=is_training, name='bn13')
    d_l7_p = tf.layers.conv2d(d_l7_a, 128, 3, stride=2, activation=tf.nn.relu, padding='valid') # 2
    d_l7_p = tf.layers.batch_normalization(d_l7_p, training=is_training, name='bn14')

    d_l8_a_1 = tf.layers.conv2d(d_l7_p, 256, 3, activation=tf.nn.relu, padding='same')
    d_l8_a_1 = tf.layers.batch_normalization(d_l8_a_1, training=is_training, name='bn15')
    d_l8_a_2 = tf.layers.conv2d(d_l8_a_1, 256, 3, activation=tf.nn.relu, padding='same')
    d_l8_a_2 = tf.layers.batch_normalization(d_l8_a_2, training=is_training, name='bn16')
    d_l8_p = tf.layers.conv2d(d_l8_a_2, 256, 3, stride=2, activation=tf.nn.relu, padding='valid') # 1
    d_l8_p = tf.layers.batch_normalization(d_l8_p, training=is_training, name='bn17')

    final_state = tf.concat([d_l8_p, ca, tl], -1)
    pre_latent = tf.layers.dense(final_state, 512, activation=tf.nn.relu)
    latent = tf.layers.dense(pre_latent, 512, activation=tf.nn.relu)

    u_l8_a = tf.layers.conv2d_transpose(latent, 512, 3, 2, activation=tf.nn.relu, padding='same') # 2
    u_l8_c = tf.concat([u_l8_a, d_l8_a_1], -1)
    u_l8_s = tf.layers.conv2d(u_l8_c, 512, 3, activation=tf.nn.relu, padding='same')
    u_l8_s = tf.layers.batch_normalization(u_l8_s, training=is_training, name='bn18')

    u_l7_a = tf.layers.conv2d_transpose(u_l8_s, 512, 3, 2, activation=tf.nn.relu, padding='same') # 4
    u_l7_c = tf.concat([u_l7_a, d_l7_a], -1)
    u_l7_s = tf.layers.conv2d(u_l7_c, 512, 3, activation=tf.nn.relu, padding='same')
    u_l7_s = tf.layers.batch_normalization(u_l7_s, training=is_training, name='bn19')

    u_l6_a = tf.layers.conv2d_transpose(u_l7_s, 256, 3, 2, activation=tf.nn.relu, padding='same') # 8
    u_l6_c = tf.concat([u_l6_a, d_l6_a], -1)
    u_l6_s = tf.layers.conv2d(u_l6_c, 256, 3, activation=tf.nn.relu, padding='same')
    u_l6_s = tf.layers.batch_normalization(u_l6_s, training=is_training, name='bn20')

    u_l5_a = tf.layers.conv2d_transpose(u_l6_s, 128, 3, 2, activation=tf.nn.relu, padding='same') # 16
    u_l5_c = tf.concat([u_l5_a, d_l5_a], -1)
    u_l5_s = tf.layers.conv2d(u_l5_c, 128, 3, activation=tf.nn.relu, padding='same')
    u_l5_s = tf.layers.batch_normalization(u_l5_s, training=is_training, name='bn21')

    u_l4_a = tf.layers.conv2d_transpose(u_l5_s, 64, 3, 2, activation=tf.nn.relu, padding='same') # 32
    u_l4_c = tf.concat([u_l4_a, d_l4_a], -1)
    u_l4_s = tf.layers.conv2d(u_l4_c, 64, 3, activation=tf.nn.relu, padding='same')
    u_l4_s = tf.layers.batch_normalization(u_l4_s, training=is_training, name='bn22')

    u_l3_a = tf.layers.conv2d_transpose(u_l4_s, 32, 3, 2, activation=tf.nn.relu, padding='same') # 64
    u_l3_c = tf.concat([u_l3_a, d_l3_a], -1)
    u_l3_s = tf.layers.conv2d(u_l3_c, 32, 3, activation=tf.nn.relu, padding='same')
    u_l3_s = tf.layers.batch_normalization(u_l3_s, training=is_training, name='bn23')

    u_l2_a = tf.layers.conv2d_transpose(u_l3_s, 32, 3, 2, activation=tf.nn.relu, padding='same') # 128
    u_l2_c = tf.concat([u_l2_a, d_l2_a_1], -1)
    u_l2_s_1 = tf.layers.conv2d(u_l2_c, 128, 5, activation=tf.nn.relu, padding='same')
    u_l2_s_1 = tf.layers.batch_normalization(u_l2_s_1, training=is_training, name='bn24')
    u_l2_s_2 = tf.layers.conv2d(u_l2_s_1, 64, 5, activation=tf.nn.relu, padding='same')
    u_l2_s_2 = tf.layers.batch_normalization(u_l2_s_2, training=is_training, name='bn25')
    u_l2_s_3 = tf.layers.conv2d(u_l2_s_2, 32, 5, activation=tf.nn.relu, padding='same')
    u_l2_s_3 = tf.layers.batch_normalization(u_l2_s_3, training=is_training, name='bn26')

#    u_l1_a = tf.layers.conv2d_transpose(u_l2_s, 64, 3, 2, activation=tf.nn.relu, padding='same') # 256
#    u_l1_c = tf.concat([u_l1_a, d_l1_a], -1)
#    u_l1_s = tf.layers.conv2d(u_l1_c, 63, 3, activation=tf.nn.relu, padding='same')

    generate_logits = tf.layers.dense(latent, 1, activation=None)
    
    generate_logits = tf.squeeze(generate_logits, [1, 2])

    moves_logits = tf.layers.conv2d(u_l2_s_3, 6, 3, activation=None, padding='same')
    
    tf.add_to_collection('m_logits', moves_logits)
    tf.add_to_collection('g_logits', generate_logits)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=moves,
                                                logits=moves_logits,
                                                dim=-1)

    losses = tf.expand_dims(losses, -1)

    masked_loss = losses * my_ships

    ships_per_frame = tf.reduce_sum(my_ships, axis=[1, 2])

    frame_loss = tf.reduce_sum(masked_loss, axis=[1, 2])

    average_frame_loss = frame_loss / tf.maximum(ships_per_frame, 1e-13) # First frames have no ship
    

    generate_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=generate, logits=generate_logits)
    
    generate_losses = tf.reduce_mean(generate_losses) # TODO: do I need to add to frames before averaging?

    loss = tf.reduce_mean(average_frame_loss) + 0.05*generate_losses
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)

    return


