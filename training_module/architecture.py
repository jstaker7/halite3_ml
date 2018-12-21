import numpy as np
import tensorflow as tf

def build_model(inference=False, num_players=1, learning_rate=None, fine_tune=False):

    frames = tf.placeholder(tf.float32, [None, 128, 128, 7])
    my_player_features = tf.placeholder(tf.float32, [None, 12])
    opponent_features = tf.placeholder(tf.float32, [None, None, 3])
    my_ships = tf.placeholder(tf.uint8, [None, 128, 128, 1])
    moves = tf.placeholder(tf.uint8, [None, 128, 128, 1])
    generate = tf.placeholder(tf.float32, [None, 1])
    will_have_ship = tf.placeholder(tf.float32, [None, 128, 128, 1])
    should_construct = tf.placeholder(tf.float32, [None, 1])
    did_win = tf.placeholder(tf.float32, [None, 1])
    is_training = tf.placeholder(tf.bool)
    
    my_ships = tf.cast(my_ships, tf.float32)

    tf.add_to_collection('frames', frames)
    tf.add_to_collection('my_player_features', my_player_features)
    tf.add_to_collection('opponent_features', opponent_features)
    tf.add_to_collection('my_ships', my_ships)
    tf.add_to_collection('moves', moves)
    tf.add_to_collection('generate', generate)
    tf.add_to_collection('is_training', is_training)
    
    tf.add_to_collection('will_have_ship', will_have_ship)
    tf.add_to_collection('should_construct', should_construct)
    tf.add_to_collection('did_win', did_win)

    moves = tf.one_hot(moves, 6)

    ca = tf.layers.conv1d(opponent_features, 16, 1, activation=tf.nn.relu)
    ca = tf.layers.conv1d(ca, 16, 1, activation=tf.nn.relu)
    ca = tf.reduce_sum(ca, 1)
    tl = tf.layers.dense(my_player_features, 32, activation=tf.nn.relu)
    tl = tf.layers.dense(tl, 16, activation=tf.nn.relu)

    ca = tf.expand_dims(ca, 1)
    ca = tf.expand_dims(ca, 1)
    tl = tf.expand_dims(tl, 1)
    tl = tf.expand_dims(tl, 1)

    d_l2_a_1 = tf.layers.conv2d(frames, 32, 3, activation=tf.nn.relu, padding='same', name='c1') # 128
    d_l2_a_1 = tf.layers.batch_normalization(d_l2_a_1, training=is_training, name='bn1')

    d_l2_p = tf.layers.conv2d(d_l2_a_1, 16, 3, strides=2, activation=tf.nn.relu, padding='same', name='c4') # 64; This might be incorrect -- too agressive downsampling considering the input size
    d_l2_p = tf.layers.batch_normalization(d_l2_p, training=is_training, name='bn4')

    d_l3_a = tf.layers.conv2d(d_l2_p, 16, 3, activation=tf.nn.relu, padding='same', name='c5')
    d_l3_a = tf.layers.batch_normalization(d_l3_a, training=is_training, name='bn5')
    d_l3_p = tf.layers.conv2d(d_l3_a, 32, 3, strides=2, activation=tf.nn.relu, padding='same', name='c6') # 32
    d_l3_p = tf.layers.batch_normalization(d_l3_p, training=is_training, name='bn6')

    d_l4_a = tf.layers.conv2d(d_l3_p, 16, 3, activation=tf.nn.relu, padding='same', name='c7')
    d_l4_a = tf.layers.batch_normalization(d_l4_a, training=is_training, name='bn7')
    d_l4_p = tf.layers.conv2d(d_l4_a, 32, 3, strides=2, activation=tf.nn.relu, padding='same', name='c8') # 16
    d_l4_p = tf.layers.batch_normalization(d_l4_p, training=is_training, name='bn8')

    d_l5_a = tf.layers.conv2d(d_l4_p, 16, 3, activation=tf.nn.relu, padding='same', name='c9')
    d_l5_a = tf.layers.batch_normalization(d_l5_a, training=is_training, name='bn9')
    d_l5_p = tf.layers.conv2d(d_l5_a, 32, 3, strides=2, activation=tf.nn.relu, padding='same', name='c10') # 8
    d_l5_p = tf.layers.batch_normalization(d_l5_p, training=is_training, name='bn10')

    d_l6_a = tf.layers.conv2d(d_l5_p, 32, 3, activation=tf.nn.relu, padding='same', name='c11')
    d_l6_a = tf.layers.batch_normalization(d_l6_a, training=is_training, name='bn11')
    d_l6_p = tf.layers.conv2d(d_l6_a, 64, 3, strides=2, activation=tf.nn.relu, padding='same', name='c12') # 4
    d_l6_p = tf.layers.batch_normalization(d_l6_p, training=is_training, name='bn12')

    d_l7_a = tf.layers.conv2d(d_l6_p, 64, 3, activation=tf.nn.relu, padding='same', name='c13')
    d_l7_a = tf.layers.batch_normalization(d_l7_a, training=is_training, name='bn13')
    d_l7_p = tf.layers.conv2d(d_l7_a, 128, 3, strides=2, activation=tf.nn.relu, padding='same', name='c14') # 2
    d_l7_p = tf.layers.batch_normalization(d_l7_p, training=is_training, name='bn14')

    d_l8_a_2 = tf.layers.conv2d(d_l7_p, 64, 3, activation=tf.nn.relu, padding='same', name='c16')
    d_l8_a_2 = tf.layers.batch_normalization(d_l8_a_2, training=is_training, name='bn16')
    d_l8_p = tf.layers.conv2d(d_l8_a_2, 128, 3, strides=2, activation=tf.nn.relu, padding='same', name='c17') # 1
    d_l8_p = tf.layers.batch_normalization(d_l8_p, training=is_training, name='bn17')

    final_state = tf.concat([d_l8_p, ca, tl], -1)
    latent = tf.layers.dense(final_state, 256, activation=tf.nn.relu, name='c19')

    u_l8_a = tf.layers.conv2d_transpose(latent, 128, 3, 2, activation=tf.nn.relu, padding='same', name='c20') # 2
    u_l8_c = tf.concat([u_l8_a, d_l8_a_2], -1)
    u_l8_s = tf.layers.conv2d(u_l8_c, 128, 3, activation=tf.nn.relu, padding='same', name='c21')
    u_l8_s = tf.layers.batch_normalization(u_l8_s, training=is_training, name='bn18')

    u_l7_a = tf.layers.conv2d_transpose(u_l8_s, 128, 3, 2, activation=tf.nn.relu, padding='same', name='c22') # 4
    u_l7_c = tf.concat([u_l7_a, d_l7_a], -1)
    u_l7_s = tf.layers.conv2d(u_l7_c, 128, 3, activation=tf.nn.relu, padding='same', name='c23')
    u_l7_s = tf.layers.batch_normalization(u_l7_s, training=is_training, name='bn19')

    u_l6_a = tf.layers.conv2d_transpose(u_l7_s, 64, 3, 2, activation=tf.nn.relu, padding='same', name='c24') # 8
    u_l6_c = tf.concat([u_l6_a, d_l6_a], -1)
    u_l6_s = tf.layers.conv2d(u_l6_c, 64, 3, activation=tf.nn.relu, padding='same', name='c25')
    u_l6_s = tf.layers.batch_normalization(u_l6_s, training=is_training, name='bn20')

    u_l5_a = tf.layers.conv2d_transpose(u_l6_s, 64, 3, 2, activation=tf.nn.relu, padding='same', name='c26') # 16
    u_l5_c = tf.concat([u_l5_a, d_l5_a], -1)
    u_l5_s = tf.layers.conv2d(u_l5_c, 64, 3, activation=tf.nn.relu, padding='same', name='c27')
    u_l5_s = tf.layers.batch_normalization(u_l5_s, training=is_training, name='bn21')

    u_l4_a = tf.layers.conv2d_transpose(u_l5_s, 64, 3, 2, activation=tf.nn.relu, padding='same', name='c28') # 32
    u_l4_c = tf.concat([u_l4_a, d_l4_a], -1)
    u_l4_s = tf.layers.conv2d(u_l4_c, 64, 3, activation=tf.nn.relu, padding='same', name='c29')
    u_l4_s = tf.layers.batch_normalization(u_l4_s, training=is_training, name='bn22')

    u_l3_a = tf.layers.conv2d_transpose(u_l4_s, 64, 3, 2, activation=tf.nn.relu, padding='same', name='30') # 64
    u_l3_c = tf.concat([u_l3_a, d_l3_a], -1)
    u_l3_s = tf.layers.conv2d(u_l3_c, 64, 3, activation=tf.nn.relu, padding='same', name='c31')
    u_l3_s = tf.layers.batch_normalization(u_l3_s, training=is_training, name='bn23')

    u_l2_a = tf.layers.conv2d_transpose(u_l3_s, 64, 3, 2, activation=tf.nn.relu, padding='same', name='c32') # 128
    u_l2_c = tf.concat([u_l2_a, d_l2_a_1], -1)
    u_l2_s_1 = tf.layers.conv2d(u_l2_c, 128, 3, activation=tf.nn.relu, padding='same', name='c33')
    u_l2_s_1 = tf.layers.batch_normalization(u_l2_s_1, training=is_training, name='bn24')
    u_l2_s_2 = tf.layers.conv2d(u_l2_s_1, 128, 3, activation=tf.nn.relu, padding='same', name='c34')
    u_l2_s_2 = tf.layers.batch_normalization(u_l2_s_2, training=is_training, name='bn25')
    
    player_generate_logits = []
    player_move_logits = []
    player_will_have_ship_logits = []
    player_should_construct_logits = []
    player_did_win_logits = []
    for i in range(num_players):

        gen_latent1 = tf.layers.dense(latent, 128, activation=tf.nn.relu, name='c39_{}'.format(i))
        gen_latent = tf.layers.dense(gen_latent1, 128, activation=tf.nn.relu, name='c39b_{}'.format(i))
        generate_logits = tf.layers.dense(gen_latent, 1, activation=None, name='c40_{}'.format(i))
        generate_logits = tf.squeeze(generate_logits, [1, 2])

        moves_latent = tf.layers.conv2d(u_l2_s_2, 64, 3, activation=tf.nn.relu, padding='same', name='c41_{}'.format(i)) # Try 1 kernel
        moves_logits = tf.layers.conv2d(moves_latent, 6, 3, activation=None, padding='same', name='c42_{}'.format(i)) # Try 1 kernel
        
        should_construct_latent = tf.layers.dense(latent, 64, activation=tf.nn.relu, name='c43_{}'.format(i))
        should_construct_logits = tf.layers.dense(should_construct_latent, 1, activation=None, name='c44_{}'.format(i))
        should_construct_logits = tf.squeeze(should_construct_logits, [1, 2])
        
        will_have_ship_latent = tf.layers.conv2d(u_l2_s_2, 64, 3, activation=tf.nn.relu, padding='same', name='c45_{}'.format(i)) # Try 1 kernel
        will_have_ship_logits = tf.layers.conv2d(will_have_ship_latent, 1, 3, activation=None, padding='same', name='c46_{}'.format(i)) # Try 1 kernel
        
        did_win_latent1 = tf.layers.dense(latent, 128, activation=tf.nn.relu, name='c47_{}'.format(i))
        did_win_latent = tf.layers.dense(did_win_latent1, 128, activation=tf.nn.relu, name='c47b_{}'.format(i))
        did_win_logits = tf.layers.dense(did_win_latent, 1, activation=None, name='c48_{}'.format(i))
        did_win_logits = tf.squeeze(did_win_logits, [1, 2])
    
        player_generate_logits.append(generate_logits)
        player_move_logits.append(moves_logits)
    
        player_will_have_ship_logits.append(will_have_ship_logits)
        player_should_construct_logits.append(should_construct_logits)
        player_did_win_logits.append(did_win_logits)

    tf.add_to_collection('m_logits', tf.stack(player_move_logits))
    tf.add_to_collection('g_logits', tf.stack(player_generate_logits))
    tf.add_to_collection('latent', latent)
    tf.add_to_collection('m_probs', tf.nn.softmax(tf.stack(player_move_logits)))
    
    tf.add_to_collection('h_logits', tf.stack(player_will_have_ship_logits))
    tf.add_to_collection('b_logits', tf.stack(player_should_construct_logits))
    tf.add_to_collection('w_logits', tf.stack(player_did_win_logits))
    
#    if inference:
#        assert False
#        return

    # TODO: Can be improved with gather_nd
    moves_logits = [tf.split(x, num_players) for x in player_move_logits]
    generate_logits = [tf.split(x, num_players) for x in player_generate_logits]
    will_have_ship_logits = [tf.split(x, num_players) for x in player_will_have_ship_logits]
    should_construct_logits = [tf.split(x, num_players) for x in player_should_construct_logits]
    did_win_logits = [tf.split(x, num_players) for x in player_did_win_logits]

    moves_logits = [x[i] for x, i in zip(moves_logits, range(num_players))]
    generate_logits = [x[i] for x, i in zip(generate_logits, range(num_players))]
    will_have_ship_logits = [x[i] for x, i in zip(will_have_ship_logits, range(num_players))]
    should_construct_logits = [x[i] for x, i in zip(should_construct_logits, range(num_players))]
    did_win_logits = [x[i] for x, i in zip(did_win_logits, range(num_players))]

    moves_logits = tf.concat(moves_logits, 0)
    generate_logits = tf.concat(generate_logits, 0)

    will_have_ship_logits = tf.concat(will_have_ship_logits, 0)
    should_construct_logits = tf.concat(should_construct_logits, 0)
    did_win_logits = tf.concat(did_win_logits, 0)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=moves,
                                                logits=moves_logits,
                                                dim=-1)

    have_ship_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=will_have_ship,
                                                logits=will_have_ship_logits)

    losses = tf.expand_dims(losses, -1)

    if True:
        kernel = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                 [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 1], [0, 0, 0]]]
                 
        kernel = np.transpose(kernel, (1, 2, 0))

        kernel = np.expand_dims(kernel, -2)

        kernel = tf.convert_to_tensor(kernel, np.float32)

        have_ship_mask = tf.nn.conv2d(my_ships, kernel, [1, 1, 1, 1], 'SAME')
        have_ship_mask = tf.stop_gradient(have_ship_mask)
    else:
        have_ship_mask = tf.constant(np.ones((1, 128, 128, 5)))

    have_ship_mask = tf.reduce_sum(have_ship_mask, -1) #> 0.5
    have_ship_mask = tf.greater(have_ship_mask, 0.5)

    have_ship_mask = tf.cast(have_ship_mask, 'float32')

    have_ship_mask = tf.expand_dims(have_ship_mask, -1)

    masked_loss = losses * my_ships
#    print(have_ship_losses.get_shape())
#    print(have_ship_mask.get_shape())
#    dsffs
    # TODO: Do a serious look at all the shapes and ensure things are proper
    have_ship_losses = have_ship_losses * have_ship_mask

    ships_per_frame = tf.reduce_sum(my_ships, axis=[1, 2])

    ship_positions_per_frame = tf.reduce_sum(have_ship_mask, axis=[1, 2])

    frame_loss = tf.reduce_sum(masked_loss, axis=[1, 2])

    have_ship_frame_loss = tf.reduce_sum(have_ship_losses, axis=[1, 2])

    average_frame_loss = frame_loss / tf.maximum(ships_per_frame, 1e-13) # First frames have no ship

    have_ship_average_frame_loss = have_ship_frame_loss / tf.maximum(ship_positions_per_frame, 1e-13) # First frames have no ship

    generate_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=generate, logits=generate_logits)
    should_construct_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=should_construct, logits=should_construct_logits)
    did_win_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=did_win, logits=did_win_logits)

    # Individual losses for validation
    player_gen_losses = [tf.reduce_mean(x) for x in tf.split(generate_losses, num_players)]
    player_should_construct_losses = [tf.reduce_mean(x) for x in tf.split(should_construct_losses, num_players)]
    player_did_win_losses = [tf.reduce_mean(x) for x in tf.split(did_win_losses, num_players)]
    player_average_frame_losses = [tf.reduce_mean(x) for x in tf.split(average_frame_loss, num_players)]
    player_have_ship_average_frame_losses = [tf.reduce_mean(x) for x in tf.split(have_ship_average_frame_loss, num_players)]
    player_total_losses = [x+0.05*y+0.5*z+0.05*w+0.01*k for x,y,z,w,k in zip(player_average_frame_losses, player_gen_losses, player_have_ship_average_frame_losses, player_should_construct_losses, player_did_win_losses)]
    
    generate_losses = tf.reduce_mean(generate_losses) # TODO: do I need to add to frames before averaging?
    should_construct_losses = tf.reduce_mean(should_construct_losses) # TODO: do I need to add to frames before averaging?
    did_win_losses = tf.reduce_mean(did_win_losses) # TODO: do I need to add to frames before averaging?

    loss = tf.reduce_mean(average_frame_loss) + 0.05*generate_losses + 0.5*tf.reduce_mean(have_ship_average_frame_loss) + 0.05*should_construct_losses + 0.01 * did_win_losses

    if fine_tune:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(loss)

    else:
        vars = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'c' == v.name[0]]

        L2_loss = tf.add_n(vars) * 0.0000001

        loss += L2_loss
    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            #optimizer = tf.contrib.opt.AdamWOptimizer(0.0000001, learning_rate)


        #tf.add_to_collection('L2_loss', L2_loss)

    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)
    tf.add_to_collection('player_gen_losses', tf.stack(player_gen_losses))
    tf.add_to_collection('player_average_frame_losses', tf.stack(player_average_frame_losses))
    tf.add_to_collection('player_have_ship_average_frame_losses', tf.stack(player_have_ship_average_frame_losses))
    tf.add_to_collection('player_total_losses', tf.stack(player_total_losses))
    tf.add_to_collection('player_should_construct_losses', tf.stack(player_should_construct_losses))
    tf.add_to_collection('did_win_losses', tf.stack(player_did_win_losses))

    return


