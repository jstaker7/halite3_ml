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

            {'pname': 'reCurs3',
             'versions': [279],
             },
           
            {'pname': 'reCurs3',
             'versions': [288],
             },
           
            {'pname': 'reCurs3',
             'versions': [306],
             },
           
            {'pname': 'reCurs3',
             'versions': [297],
             },

            {'pname': 'SiestaGuru',
             'versions': [327],
             },
           
            {'pname': 'SiestaGuru',
             'versions': [312],
             },
           
            {'pname': 'SiestaGuru',
             'versions': [313],
             },
           
            {'pname': 'SiestaGuru',
             'versions': [317],
             },
]

in_train = set()
in_valid = set()

def filter_replays(pname, versions):
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

    train, valid = [], []
    new_keep = []
    for rp in keep:
        if rp in in_train:
            train.append(rp)
        elif rp in in_valid:
            valid.append(rp)
        else:
            new_keep.append(rp)

    _train, _valid = new_keep[:int(len(new_keep)/1.33)], new_keep[int(len(new_keep)/1.33):]

    train += _train
    valid += _valid

    in_train |= set(train)
    in_valid |= set(valid)

    return train, valid

for player in PLAYERS:
    train, valid = filter_replays(player['pname'], player['versions'])
    player['train'] = train
    player['valid'] = valid
    print("{} num train: {} num valid: {}".format(player['pname'], len(train), len(valid)))

del master_index

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


min_buffer_size = 2500
max_buffer_size = 4000
batch_size = 4

#min_buffer_size = 500
#max_buffer_size = 800
#batch_size = 2


def batch_prep(buffer, batch_queue):
    game = Game()
    frames = []
    moves = []
    generates = []
    my_player_features = []
    opponent_features = []
    my_ships = []
    
    will_have_ships = []
    should_constructs = []
    did_wins = []

    while True:
        if buffer.qsize() < min_buffer_size:
            time.sleep(1)
            continue
        p, t, data = buffer.get()
        
        frame, move, generate, my_player_feature, opponent_feature, will_have_ship, should_construct, did_win = data
        frame = np.expand_dims(frame, 0)
        move = np.expand_dims(move, 0)
        will_have_ship = np.expand_dims(will_have_ship, 0)
        frame, my_ship, move, will_have_ship = game.pad_replay(frame, move, will_have_ship=will_have_ship)

        frames.append(frame[0])
        moves.append(move[0])
        will_have_ships.append(will_have_ship[0])
        generates.append(generate)
        my_player_features.append(my_player_feature)
        opponent_features.append(opponent_feature)
        my_ships.append(my_ship[0])
        should_constructs.append(should_construct)
        did_wins.append(did_win)

        if len(frames) == batch_size:
            pair = np.array(frames), np.array(moves), np.array(generates), np.array(my_player_features), np.array(opponent_features), np.array(my_ships), np.array(will_have_ships), np.array(should_constructs), np.array(did_wins)
            batch_queue.put(copy.deepcopy(pair))
            
            del pair
            del frames
            del moves
            del generates
            del my_player_features
            del opponent_features
            del my_ships
            del will_have_ships
            del should_constructs
            del did_wins
            
            # Reset
            frames = []
            moves = []
            generates = []
            my_player_features = []
            opponent_features = []
            my_ships = []
            will_have_ships = []
            should_constructs = []
            did_wins = []


def buffer(raw_queues, buffer_q):
    while True:
        which_queue = np.random.randint(5)
        queue = raw_queues[which_queue]

        pair = queue.get()
        
        # Basically shuffles as it goes
        rand_priority = np.random.randint(max_buffer_size)
        buffer_q.put((rand_priority, time.time(), pair))

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
    
        frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win = game.get_training_frames(pname=pname)
        
        # Avoid GC issues
        frames = copy.deepcopy(frames)
        moves = copy.deepcopy(moves)
        generate = copy.deepcopy(generate)
        my_player_features = copy.deepcopy(my_player_features)
        opponent_features = copy.deepcopy(opponent_features)
        will_have_ship = copy.deepcopy(will_have_ship)
        should_construct = copy.deepcopy(should_construct)
        did_win = copy.deepcopy(did_win)
        del game
        
#        frames = frames[:25]
#        moves = moves[:25]
#        generate = generate[:25]
#        can_afford = can_afford[:25]
#        turns_left = turns_left[:25]

        shapes = [frames.shape[0], moves.shape[0], generate.shape[0], my_player_features.shape[0], opponent_features.shape[0], will_have_ship.shape[0], should_construct.shape[0], did_win.shape[0]]

        assert len(set(shapes)) == 1, print(shapes)

        for pair in zip(frames, moves, generate, my_player_features, opponent_features, will_have_ship, should_construct, did_win):
            #buffer.append(pair)
            queue.put(copy.deepcopy(pair))

        del frames
        del moves
        del generate
        del my_player_features
        del opponent_features
        del will_have_ship
        del should_construct
        del did_win
    
#        if len(buffer) > 10:
#            shuffle(buffer)
#            while len(buffer) > 0:
#                queue.put(buffer.pop())
        #queue.put(size)

processes = []
for player in PLAYERS:

    # 5 queues, 1 for each map size (to improve compute efficiency)
    queues = [Queue(2) for _ in range(5)]
    queue_m_sizes = [32, 40, 48, 56, 64]

    v_queues = [Queue(2) for _ in range(5)]
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

learning_rate = tf.placeholder(tf.float32, shape=[])

build_model(num_players=len(PLAYERS), learning_rate=learning_rate)

frames_node = tf.get_collection('frames')[0]
opponent_features_node = tf.get_collection('opponent_features')[0]
my_player_features_node = tf.get_collection('my_player_features')[0]
my_ships_node = tf.get_collection('my_ships')[0]
moves_node = tf.get_collection('moves')[0]
generate_node = tf.get_collection('generate')[0]
loss_node = tf.get_collection('loss')[0]
optimizer_node = tf.get_collection('optimizer')[0]
is_training = tf.get_collection('is_training')[0]

player_gen_losses_node = tf.get_collection('player_gen_losses')[0]
player_average_frame_losses_node = tf.get_collection('player_average_frame_losses')[0]
player_total_losses_node = tf.get_collection('player_total_losses')[0]
player_have_ship_average_frame_losses_node = tf.get_collection('player_have_ship_average_frame_losses')[0]
player_should_construct_losses_node = tf.get_collection('player_should_construct_losses')[0]
did_win_losses_node = tf.get_collection('did_win_losses')[0]

will_have_ship_node = tf.get_collection('will_have_ship')[0]
should_construct_node = tf.get_collection('should_construct')[0]
did_win_node = tf.get_collection('did_win')[0]
#L2_loss_node = tf.get_collection('L2_loss')[0]



#with open() # TODO: save out log

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config=tf.ConfigProto(gpu_options=gpu_options)
saver = tf.train.Saver(max_to_keep=None)
#best = np.ones((len(PLAYERS), ), dtype=np.int32)*999
best = np.array([999 for _ in range(len(PLAYERS))])
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
        reg_losses = []
        for step in range(20000000):
            player_batches = []
            for player in PLAYERS:
                batch = player['batch_q'].get()
                player_batches.append(batch)
            
            batch = [np.concatenate(x, 0) for x in zip(*player_batches)]
            shapes = [x.shape[0] for x in batch]
            assert len(set(shapes)) == 1
            f_batch, m_batch, g_batch, c_batch, t_batch, s_batch, h_batch, b_batch, w_batch = batch

            g_batch = np.expand_dims(g_batch, -1)
            m_batch = np.expand_dims(m_batch, -1)
            s_batch = np.expand_dims(s_batch, -1)
            h_batch = np.expand_dims(h_batch, -1)
            b_batch = np.expand_dims(b_batch, -1)
            w_batch = np.expand_dims(w_batch, -1)
            

            #print([x.shape for x in [f_batch, m_batch, g_batch, c_batch, t_batch, s_batch]])
            
            #print(np.sum(s_batch))
            
            T = 400000
            M = 10#20#2 #T/20000
            t = step
            lr = (0.0003)*(np.cos(np.pi*np.mod(t - 1, T/M)/(T/M)) + 1)

            feed_dict = {frames_node: f_batch,
                         my_player_features_node: c_batch,
                         opponent_features_node: t_batch,
                         my_ships_node: s_batch,
                         moves_node: m_batch,
                         generate_node: g_batch,
                         is_training: True,
                         learning_rate: lr,
                         will_have_ship_node: h_batch,
                         should_construct_node: b_batch,
                         did_win_node: w_batch
                        }

#            temp_node = tf.get_collection('temp')[0]
#            print(f_batch[0, 55:65, 55:65, 1])
#            print(m_batch[0, 55:65, 55:65, 0])
#            temp = sess.run([temp_node], feed_dict=feed_dict)
#            print(temp[0][0, 55:65, 55:65])
#            print('fannty')
#            import time
#            time.sleep(1)
#            continue

            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
            reg_loss = 0
            losses.append(loss)
            reg_losses.append(reg_loss)
            if (step + 1) % 1250 == 0 or step == 0:
                player_gen_losses = []
                player_average_frame_losses = []
                player_total_losses = []
                player_have_ship_losses = []
                player_should_construct_losses = []
                player_did_win_losses = []
                for vstep in range(3000):
                
                    player_batches = []
                    for player in PLAYERS:
                        batch = player['v_batch_q'].get()
                        player_batches.append(batch)
                
                    batch = [np.concatenate(x, 0) for x in zip(*player_batches)]
                    f_batch, m_batch, g_batch, c_batch, t_batch, s_batch, h_batch, b_batch, w_batch = batch
    
                    g_batch = np.expand_dims(g_batch, -1)
                    m_batch = np.expand_dims(m_batch, -1)
                    s_batch = np.expand_dims(s_batch, -1)
                    h_batch = np.expand_dims(h_batch, -1)
                    b_batch = np.expand_dims(b_batch, -1)
                    w_batch = np.expand_dims(w_batch, -1)

                    feed_dict = {frames_node: f_batch,
                                 my_player_features_node: c_batch,
                                 opponent_features_node: t_batch,
                                 my_ships_node: s_batch,
                                 moves_node: m_batch,
                                 generate_node: g_batch,
                                 is_training: False,
                                 will_have_ship_node: h_batch,
                                 should_construct_node: b_batch,
                                 did_win_node: w_batch
                                }

                    gen_loss, frame_loss, total_loss, hs_loss, b_loss, w_loss = sess.run([player_gen_losses_node, player_average_frame_losses_node, player_total_losses_node, player_have_ship_average_frame_losses_node, player_should_construct_losses_node, did_win_losses_node], feed_dict=feed_dict)
                    player_gen_losses.append(gen_loss)
                    player_average_frame_losses.append(frame_loss)
                    player_total_losses.append(total_loss)
                    
                    player_have_ship_losses.append(hs_loss)
                    player_should_construct_losses.append(b_loss)
                    player_did_win_losses.append(w_loss)
                    
                    if step == 0 and vstep == 5:
                        break
            
                player_gen_losses = np.stack(player_gen_losses, 1)
                player_average_frame_losses = np.stack(player_average_frame_losses, 1)
                player_total_losses = np.stack(player_total_losses, 1)
                player_have_ship_losses = np.stack(player_have_ship_losses, 1)
                player_should_construct_losses = np.stack(player_should_construct_losses, 1)
                player_did_win_losses = np.stack(player_did_win_losses, 1)
                
                player_gen_losses = np.mean(player_gen_losses, 1)
                player_average_frame_losses = np.mean(player_average_frame_losses, 1)
                player_total_losses = np.mean(player_total_losses, 1)
                player_have_ship_losses = np.mean(player_have_ship_losses, 1)
                player_should_construct_losses = np.mean(player_should_construct_losses, 1)
                player_did_win_losses = np.mean(player_did_win_losses, 1)
                
                assert player_total_losses.shape[0] == len(PLAYERS)
                
                #new_losses = []
                
                player_print = " ".join(["{:.3f}/{:.3f}".format(x,y) for x,y in zip(player_average_frame_losses, player_gen_losses)])
                
                print_line = "{} T: {:.3f} V: ".format(step, np.mean(losses[-1000:])) + player_print
            
                if np.sum(np.less(player_total_losses, best)) == len(PLAYERS): # All players must have improved
                    best = player_total_losses
                    saver.save(sess, os.path.join(save_dir, 'model_{}.ckpt'.format(step)))
                    print(print_line + " *** new best ***")
                else:
                    print(print_line)

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
