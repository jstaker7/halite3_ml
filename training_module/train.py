import pickle
import gzip
import time
from random import shuffle
#from multiprocessing import Process, Queue
import os
from threading import Thread
from queue import Queue, PriorityQueue
import copy

import numpy as np
import tensorflow as tf

from core.data_utils import Game

from training_module.architecture import build_model

np.random.seed(8)

RESTORE = False#True
RESTORE_WHICH = '4'

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

save_dir = '/home/staker/Projects/halite/models/'

if not os.path.exists(save_dir):
    save_dir = '/Users/Peace/Desktop/model_temp'

if not os.path.exists(replay_root):
    replay_root = '/home/staker/Projects/halite/replays'

with gzip.open(os.path.join(replay_root, 'INDEX.pkl'), 'rb') as infile:
    master_index = pickle.load(infile)

PLAYERS = [
            {'pname': 'TheDuck314',
             'versions': [30, 31, 33, 34, 35, 36], # 39, 40, 41, 42, 43, 44
             },
           
            {'pname': 'teccles',
             'versions': [96, 97, 98, 99, 100, 101, 102, 103],
             },

            {'pname': 'reCurs3',
             'versions': [113, 114, 115, 117, 120],
             },
]

in_train = set()
in_valid = set()

def filter_replays(pname, versions, DUCK):
    global in_train
    global in_valid
    keep = []
    for rp in master_index:
        for p in master_index[rp]['players']:
            name, _version = p['name'].split(' v')
            version = int(_version.strip())
            if pname == name and version in versions:
                keep.append(rp)
                break

    if DUCK:
        np.random.shuffle(keep) # in place

        keep, valid = keep[:len(keep)//2], keep[len(keep)//2:]
        in_train |= set(keep)
        in_valid |= set(valid)
        return keep, valid

    else:
        train, valid = [], []
        new_keep = []
        for rp in keep:
            if rp in in_train:
                train.append(rp)
            elif rp in in_valid:
                valid.append(rp)
            else:
                new_keep.append(rp)

        _train, _valid = new_keep[:len(new_keep)//2], new_keep[len(new_keep)//2:]

        train += _train
        valid += _valid

        in_train |= set(train)
        in_valid |= set(valid)

        return train, valid

for player in PLAYERS:
    train, valid = filter_replays(player['pname'], player['versions'], player['pname'] == 'TheDuck314')
    player['train'] = train
    player['valid'] = valid
    print(player['pname'])
    print(len(train))
    print(len(valid))

#assert keep, print(len(keep))

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

# NEXT: Fix workers


min_buffer_size = 5000
max_buffer_size = 8000
batch_size = 16


def batch_prep(buffer, batch_queue):
    game = Game()
    frames = []
    moves = []
    generates = []
    can_affords = []
    turns_lefts = []
    my_ships = []

    while True:
        if buffer.qsize() < min_buffer_size:
            time.sleep(1)
            continue
        p, t, data = buffer.get()
        #print('a')
        #print(p)
        #print(t)
        
        frame, move, generate, can_afford, turns_left = data
        frame = np.expand_dims(frame, 0)
        move = np.expand_dims(move, 0)
        frame, my_ship, move = game.pad_replay(frame, move)
        
        #print(np.sum(my_ship))
        #time.sleep(1)

        frames.append(frame[0])
        moves.append(move[0])
        generates.append(generate)
        can_affords.append(can_afford)
        turns_lefts.append(turns_left)
        my_ships.append(my_ship[0])

        if len(frames) == batch_size:
            pair = np.array(frames), np.array(moves), np.array(generates), np.array(can_affords), np.array(turns_lefts), np.array(my_ships)
            batch_queue.put(copy.deepcopy(pair))
            
            del pair
            del frames
            del moves
            del generates
            del can_affords
            del turns_lefts
            del my_ships
            
            # Reset
            frames = []
            moves = []
            generates = []
            can_affords = []
            turns_lefts = []
            my_ships = []



def buffer(raw_queues, buffer_q):
    while True:
        which_queue = np.random.randint(5)
        queue = raw_queues[which_queue]

        pair = queue.get()
        
        #for pair in zip(game):
        # Basically shuffles as it goes
        rand_priority = np.random.randint(max_buffer_size)
        #print('b')
        #print(rand_priority)
        buffer_q.put((rand_priority, time.time(), pair))
        #time.sleep(1)

def worker(queue, size, pname, keep):
    np.random.seed(size) # Use size as seed
    # Filter out games that are not the right size
    # Note: Replay naming is not consistent (game id was added later)
    s_keep = [x for x in keep if int(x.split('-')[-2]) == size]
    print("{0} {1} maps with size {2}x{2}".format(pname, len(s_keep), size))
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
        
        frames, moves, generate, can_afford, turns_left = game.get_training_frames(pname=pname)
        
        # Avoid GC issues
        frames = copy.deepcopy(frames)
        moves = copy.deepcopy(moves)
        generate = copy.deepcopy(generate)
        can_afford = copy.deepcopy(can_afford)
        turns_left = copy.deepcopy(turns_left)
        del game
        
#        frames = frames[:25]
#        moves = moves[:25]
#        generate = generate[:25]
#        can_afford = can_afford[:25]
#        turns_left = turns_left[:25]

        for pair in zip(frames, moves, generate, can_afford, turns_left):
            #buffer.append(pair)
            queue.put(copy.deepcopy(pair))

        del frames
        del moves
        del generate
        del can_afford
        del turns_left
    
#        if len(buffer) > 10:
#            shuffle(buffer)
#            while len(buffer) > 0:
#                queue.put(buffer.pop())
        #queue.put(size)

processes = []
for player in PLAYERS:

    # 5 queues, 1 for each map size (to improve compute efficiency)
    queues = [Queue(32) for _ in range(5)]
    queue_m_sizes = [32, 40, 48, 56, 64]

    v_queues = [Queue(32) for _ in range(5)]
    v_queue_m_sizes = [32, 40, 48, 56, 64]

    batch_queue = Queue(2)
    buffer_queue = PriorityQueue(max_buffer_size)

    v_batch_queue = Queue(2)
    v_buffer_queue = PriorityQueue(max_buffer_size)

    processes += [Thread(target=worker, args=(queues[ix], queue_m_sizes[ix], player['pname'], player['train'])) for ix in range(5)]
    processes += [Thread(target=worker, args=(v_queues[ix], v_queue_m_sizes[ix], player['pname'], player['valid'])) for ix in range(5)]

    buffer_thread = Thread(target=buffer, args=(queues, buffer_queue))
    batch_thread = Thread(target=batch_prep, args=(buffer_queue, batch_queue))

    v_buffer_thread = Thread(target=buffer, args=(v_queues, v_buffer_queue))
    v_batch_thread = Thread(target=batch_prep, args=(v_buffer_queue, v_batch_queue))

    processes += [buffer_thread, batch_thread, v_buffer_thread, v_batch_thread]

    player['batch_q'] = batch_queue
    player['v_batch_q'] = v_batch_queue

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
is_training = tf.get_collection('is_training')[0]

#with open() # TODO: save out log

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
config=tf.ConfigProto(gpu_options=gpu_options)
saver = tf.train.Saver()
best = 999
try:
    with tf.Session(config=config) as sess:
        tf.initializers.global_variables().run()
        
        if RESTORE:
            print("Restoring...")
            saver.restore(sess, os.path.join('/home/staker/Projects/halite/trained_models/', RESTORE_WHICH, "model.ckpt"))
        
        # Training buffer to decorrelate examples seen in batches
    #    buffer = []
    #
    #
    #
    #    assert len(buffer) > batch_size

        print("Training...")
        losses = []
        for step in range(20000000):
            player_batches = []
            for player in PLAYERS:
                batch = player['batch_q'].get()
                player_batches.append(batch)
            
            if len(PLAYERS) == 1:
                f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = batch
            else:
                batch = [np.concatenate(x, 0) for x in zip(*player_batches)]
                f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = batch
            
            #f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = data

            #f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = [], [], [], [], [], []
            
    #        #shuffle(buffer)
    #
    #        for i in range(batch_size):
    #            frame, move, generate, can_afford, turns_left, my_ships = buffer.pop()
    #            f_batch.append(frame)
    #            m_batch.append(move)
    #            g_batch.append(generate)
    #            c_batch.append(can_afford)
    #            t_batch.append(turns_left)
    #            s_batch.append(my_ships)
    #
    #        # Replenish buffer
    #        for i in range(batch_size):
    #            which_queue = np.random.randint(5)
    #            queue = queues[which_queue]
    #            pair = queue.get()
    #            buffer.append(pair)
    #
    #        f_batch = np.stack(f_batch)
    #        m_batch = np.stack(m_batch)
    #        g_batch = np.stack(g_batch)
    #        c_batch = np.stack(c_batch)
    #        t_batch = np.stack(t_batch)
    #        s_batch = np.stack(s_batch)
    #
    #        buffer_queue

            g_batch = np.expand_dims(g_batch, -1)
            t_batch = np.expand_dims(t_batch, -1)
            m_batch = np.expand_dims(m_batch, -1)
            s_batch = np.expand_dims(s_batch, -1)
            
            #print([x.shape for x in [f_batch, m_batch, g_batch, c_batch, t_batch, s_batch]])
            
            #print(np.sum(s_batch))
            
            feed_dict = {frames_node: f_batch,
                         can_afford_node: c_batch,
                         turns_left_node: t_batch,
                         my_ships_node: s_batch,
                         moves_node: m_batch,
                         generate_node: g_batch,
                         is_training: True
                        }

            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
            losses.append(loss)
            if step % 5000 == 0:
                v_losses = []
                for _ in range(3000):
                
                    player_batches = []
                    for player in PLAYERS:
                        batch = player['v_batch_q'].get()
                        player_batches.append(batch)
                    
                    if len(PLAYERS) == 1:
                        f_batch, m_batch, g_batch, c_batch, t_batch, s_batch = batch
                    else:
                        pass # Combine them into a single batch
    
                    g_batch = np.expand_dims(g_batch, -1)
                    t_batch = np.expand_dims(t_batch, -1)
                    m_batch = np.expand_dims(m_batch, -1)
                    s_batch = np.expand_dims(s_batch, -1)

                    feed_dict = {frames_node: f_batch,
                                 can_afford_node: c_batch,
                                 turns_left_node: t_batch,
                                 my_ships_node: s_batch,
                                 moves_node: m_batch,
                                 generate_node: g_batch,
                                 is_training: False
                                }

                    loss = sess.run(loss_node, feed_dict=feed_dict)
                    v_losses.append(loss)
            
                if np.mean(v_losses) < best:
                    best = np.mean(v_losses)
                    #saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
                    print("{} {:.2f} {:.2f} *** new best ***".format(step, np.mean(losses[-1000:]), np.mean(v_losses)))
                else:
                    print("{} {:.2f} {:.2f}".format(step, np.mean(losses[-1000:]), np.mean(v_losses)))

    #        for i in range(100000):
    #            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
    #            print(loss)

            #val = queue.get()
            #print(val)

    # Probably want to mean by ship number and weight other factors, like time
    # step in game to balance the training.

except KeyboardInterrupt:
    print('Cleaning up')
    [p.join() for p in processes]
