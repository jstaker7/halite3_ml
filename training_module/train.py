import pickle
import gzip
import time
from random import shuffle
#from multiprocessing import Process, Queue
import os
from threading import Thread
from queue import Queue

import numpy as np
import tensorflow as tf

from core.data_utils import Game

from training_module.architecture import build_model

# Given history of frames
# 1) Predict which player is which
# 2) Keep a history vector
# 4)

# U-net convolve
# Latent vector gets passed into RNN to update state
# U-net deconvolve + state to make pixel predictions
# Truncated history of only X frames (to both decorrelate, and b/c that much
# histrory won't affect the bots much.
# Once a model is trained on top N players, us RL to select which player to
# base the next move off of.
# Consider pretraining the u-net (or ladder net style) with more data if needed.
# Fine-tune train as player bots get updated

#receptive_field_size = 3

# Ensure that the frame wraps along edges

# Potentially learn a reconstruction error to capture many states (of gold only)

# Start with greedy (single convolution)

# load the replay index
#with gzip.open('/Users/Peace/Desktop/replays/INDEX.pkl', 'rb') as infile:
#    master_index = pickle.load(infile)
#
#print(master_index.keys())

replay_root = '/Users/Peace/Desktop/replays'

if not os.path.exists(replay_root):
    replay_root = '/home/staker/Projects/halite/replays'

with gzip.open(os.path.join(replay_root, 'INDEX.pkl'), 'rb') as infile:
    master_index = pickle.load(infile)

keep = []
for rp in master_index:
    for p in master_index[rp]['players']:
        if 'Rachol' == p['name'].split(' ')[0]:
            keep.append(rp)
            break

# Test speed before MP
#for rp in keep:
#    path = '/Users/Peace/Desktop/replays/{}/{}'
#    day = rp.replace('ts2018-halite-3-gold-replays_replay-', '').split('-')[0]
#    path = path.format(day, rp)
#
#    game = Game()
#    try:
#        game.load_replay(path)
#        frames, moves = game.get_training_frames(pname='Rachol')
#        print(frames.shape, moves.shape)
#    except:
#        print('skipped')
#        continue
#
#    print()

def worker(queue, size):
    np.random.seed(size) # Use size as seed
    # Filter out games that are not the right size
    # Note: Replay naming is not consistent (game id was added later)
    s_keep = [x for x in keep if int(x.split('-')[-2]) == size]
    print("{0} maps with size {1}x{1}".format(len(s_keep), size))
    #buffer = []
    while True:
        which_game = np.random.choice(s_keep)
        path = os.path.join(replay_root, '{}', '{}')
        day = which_game.replace('ts2018-halite-3-gold-replays_replay-', '').split('-')[0]
        path = path.format(day, which_game)

        game = Game()
        try:
            game.load_replay(path)
        except:
            continue
        
        frames, moves, generate, can_afford, turns_left, my_ships = game.get_training_frames(pname='Rachol')
        
#        frames = frames[:25]
#        moves = moves[:25]
#        generate = generate[:25]
#        can_afford = can_afford[:25]
#        turns_left = turns_left[:25]

        for pair in zip(frames, moves, generate, can_afford, turns_left, my_ships):
            #buffer.append(pair)
            queue.put(pair)
    
#        if len(buffer) > 10:
#            shuffle(buffer)
#            while len(buffer) > 0:
#                queue.put(buffer.pop())
        #queue.put(size)


# 5 queues, 1 for each map size (to improve compute efficiency)
queues = [Queue(32) for _ in range(5)]
queue_m_sizes = [32, 40, 48, 56, 64]

batch_size = 32

#processes = [Process(target=worker, args=(queues[ix], queue_m_sizes[ix])) for ix in range(5)]
processes = [Thread(target=worker, args=(queues[ix], queue_m_sizes[ix])) for ix in range(5)]
[p.start() for p in processes]

build_model()

frames_node = tf.get_collection('frames')[0]
can_afford_node = tf.get_collection('can_afford')[0]
turns_left_node = tf.get_collection('turns_left')[0]
my_ships_node = tf.get_collection('my_ships')[0]
moves_node = tf.get_collection('moves')[0]
generate_node = tf.get_collection('generate')[0]
loss_node = tf.get_collection('loss')[0]
optimizer_node = tf.get_collection('optimizer')[0]

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.initializers.global_variables().run()
    
    # Training buffer to decorrelate examples seen in batches
    buffer = []
    
    print("Filling buffer...")
    for _ in range(6000):
        which_queue = np.random.randint(5)
        queue = queues[which_queue]
        pair = queue.get()
        buffer.append(pair)
    
    assert len(buffer) > batch_size

    print("Training...")
    for step in range(2000000):

        f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = [], [], [], [], [], []
        
        shuffle(buffer)
    
        for i in range(batch_size):
            frame, move, generate, can_afford, turns_left, my_ships = buffer.pop()
            f_batch.append(frame)
            m_batch.append(move)
            g_batch.append(generate)
            c_batch.append(can_afford)
            t_batch.append(turns_left)
            s_batch.append(my_ships)

        # Replenish buffer
        for i in range(batch_size):
            which_queue = np.random.randint(5)
            queue = queues[which_queue]
            pair = queue.get()
            buffer.append(pair)
        
        f_batch = np.stack(f_batch)
        m_batch = np.stack(m_batch)
        g_batch = np.stack(g_batch)
        c_batch = np.stack(c_batch)
        t_batch = np.stack(t_batch)
        s_batch = np.stack(s_batch)
        
        g_batch = np.expand_dims(g_batch, -1)
        t_batch = np.expand_dims(t_batch, -1)
        m_batch = np.expand_dims(m_batch, -1)
        s_batch = np.expand_dims(s_batch, -1)
        
        #print([x.shape for x in [f_batch, m_batch, g_batch, c_batch, t_batch, s_batch]])
        
        feed_dict = {frames_node: f_batch,
                     can_afford_node: c_batch,
                     turns_left_node: t_batch,
                     my_ships_node: s_batch,
                     moves_node: m_batch,
                     generate_node: g_batch,
                    }

        loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)

        if step % 100 == 0:
            print(step)
            print(loss)
            saver.save(sess, '/home/staker/Projects/halite/models/model.ckpt')
        
#        for i in range(100000):
#            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
#            print(loss)

        #val = queue.get()
        #print(val)

# Probably want to mean by ship number and weight other factors, like time
# step in game to balance the training.