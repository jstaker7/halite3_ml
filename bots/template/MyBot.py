#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
#import arrow

#logger = logging.getLogger('./test.log')
#logger.setLevel(30)
#logging.info('test')

LOCAL = True

import numpy as np

# Turning off logging doesn't seem to be working
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import device_lib

#tf.logging.set_verbosity(tf.logging.WARN) # Turn this on before uploading
#tf.logging.set_verbosity(0)
#tf.logging.set_verbosity()

cd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cd)
from data_utils import center_frames, pad_replay

# Required inputs:
# production, has_ship, entity_energies, factories, has_dropoff
# can_afford
# turns_left

MAP_SIZES = [32, 40, 48, 56, 64]

opponents = {}

def update_frame(frame, num_players, my_id, map_dim, max_turns):

    global opponents

    # Some channels get refreshed entirely
    frame[:, :, [1,2,4,5,6]] = 0

    turn_number = input()
    
    turns_left_raw = float(max_turns) - float(turn_number)
    
    turns_left = (turns_left_raw)/200. - 1. # TODO: Is this off by one?
    
    if LOCAL:
        logging.info(turns_left_raw)

    turns_left = np.expand_dims(turns_left, 0)
    
    # TODO: Add in feature to account for my halite vs the opponents
    
    my_halite = None
    my_ships = None
    
    my_dropoffs = []
    enemy_dropoffs = []
    
    has_ship = np.zeros((frame.shape[1], frame.shape[1]), dtype=np.uint8)

    for _ in range(num_players):
        player, num_ships, num_dropoffs, halite = [int(x) for x in input().split()]
        
        ships = [[int(x) for x in input().split()] for _ in range(num_ships)]
        dropoffs = [[int(x) for x in input().split()] for _ in range(num_dropoffs)]
        
        if player == my_id:
            my_halite = halite
            my_ships = ships
        else:
            if player not in opponents:
                opponents[player] = {'halite': halite, 'num_ships': num_ships}
        
        
        for ship in ships:
            id, x, y, h = ship
            frame[y, x, 2] = h
            if player == my_id:
                frame[y, x, 1] = 1.
                frame[y, x, 5] = float(h > 999)
                frame[y, x, 6] = float(id)/50.
            else:
                frame[y, x, 1] = -1.
            
            has_ship[y, x] = 1
    
        for dropoff in dropoffs:
            id, x, y = dropoff
            if player == my_id:
                frame[y, x, 4] = 1.
                my_dropoffs.append((y, x))
            else:
                frame[y, x, 4] = -1.
                enemy_dropoffs.append((y, x))

    
    for _ in range(int(input())):
        x, y, h = [int(x) for x in input().split()]
        frame[y, x, 0] = h

    assert my_halite is not None

    can_afford_both = my_halite > 4999.
    can_afford_drop = my_halite > 3999.
    can_afford_ship = my_halite > 999.

    can_afford = np.stack([can_afford_ship, can_afford_drop, can_afford_both], -1)
    
    has_ship = np.expand_dims(has_ship, 0)

    op_ids = sorted(list(opponents.keys()))

    opponent_energy = np.array([opponents[x]['halite'] for x in op_ids])

    opponent_energy = opponent_energy.reshape((1, -1))

    # TODO: This only needs to be computed once
    map_size_ix = MAP_SIZES.index(map_dim)
    map_size = np.zeros((len(MAP_SIZES),), dtype=np.float32)
    map_size[map_size_ix] = 1.

    _my_halite = int(my_halite)

    my_halite = np.log10(_my_halite/1000. + 1)
    my_halite = np.expand_dims(my_halite, -1)
    enemy_halite = np.log10(opponent_energy/1000. + 1)
    _halite_diff = np.expand_dims(_my_halite, -1) - opponent_energy
    halite_diff = np.sign(_halite_diff) * np.log10(np.absolute(_halite_diff)/1000. + 1)

    num_opponents = 0 if len(op_ids) == 1 else 1

    enemy_ship_counts = np.array([opponents[x]['num_ships'] for x in op_ids])

    num_opponent_ships = enemy_ship_counts/50.
    num_opponent_ships = np.expand_dims(num_opponent_ships, -1)
    num_my_ships = [len(my_ships)/50.]
        
    meta_features = np.array(list(map_size) +  [num_opponents])

    assert meta_features.shape[0] == 6

    meta_features = np.expand_dims(meta_features, 0)
    meta_features = np.tile(meta_features, [enemy_halite.shape[0], 1])

    opponent_features = [enemy_halite, halite_diff, num_opponent_ships]
    #print([x.shape for x in opponent_features])
    opponent_features = np.stack(opponent_features, -1)
    if opponent_features.shape[1] == 1:
        opponent_features = np.pad(opponent_features, ((0,0), (0,2), (0,0)), 'constant', constant_values=0)
    my_player_features = [my_halite, turns_left, can_afford, num_my_ships]

    my_player_features = np.concatenate(my_player_features, -1)

    my_player_features = np.expand_dims(my_player_features, 0)
    #print(meta_features.shape)
    #
    my_player_features = np.concatenate([my_player_features, meta_features], -1)

    return frame, turns_left_raw, my_player_features, opponent_features, my_ships, has_ship, my_halite, my_dropoffs, enemy_dropoffs

def get_initial_data():
    raw_constants = input()

    constants = json.loads(raw_constants)
    
    max_turns = constants['MAX_TURNS'] # Only one I think we need

    num_players, my_id = [int(x) for x in input().split()]
    
    player_tups = []
    for player in range(num_players):
        p_tup = map(int, input().split())
        player_tups.append(p_tup)
    
    map_width, map_height = map(int, input().split())
    map_dim = map_width # Assuming square maps (to keep things simple)

    game_map = []
    for _ in range(map_dim):
        row = [int(x) for x in input().split()]
        game_map.append(row)

    halite = np.array(game_map)

    return max_turns, num_players, my_id, halite, player_tups, map_dim

valid_moves = ['o', 'n', 'e', 's', 'w', 'c']
move_shifts = [(0,0), (-1,0), (0,1), (1,0), (0,-1), (0,0)]

assert os.path.exists(os.path.join(cd, "model.ckpt.meta"))
assert os.path.exists(os.path.join(cd, "model.ckpt.data-00000-of-00001"))

#logging.info("Starting session...")
# Load the model
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    # Check devices
    #local_device_protos = device_lib.list_local_devices()
    with tf.device('/gpu:0'):
        tf.train.import_meta_graph(os.path.join(cd, "model.ckpt.meta"))
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(cd, "model.ckpt"))
    
    # TODO: Also load player model
    
    frames_node = tf.get_collection('frames')[0]
    my_player_features_node = tf.get_collection('my_player_features')[0]
    opponent_features_node = tf.get_collection('opponent_features')[0]
    moves_node = tf.get_collection('m_logits')[0]
    generate_node = tf.get_collection('g_logits')[0]
    is_training = tf.get_collection('is_training')[0]
    m_probs = tf.get_collection('m_probs')[0]
    h_logits = tf.get_collection('h_logits')[0]
    b_logits = tf.get_collection('b_logits')[0]
    w_logits = tf.get_collection('w_logits')[0]
    
    max_turns, num_players, my_id, halite, player_tups, map_dim = get_initial_data()

    if LOCAL:
        logging.basicConfig(filename='{}-bot.log'.format(my_id),
                                    filemode='w',
                                    #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                    #datefmt='%H:%M:%S',
                                    level=logging.DEBUG)

    frame = np.zeros((map_dim, map_dim, 7), dtype=np.float32)
    frame[:, :, 0] = halite
    
    enemy_shipyards = []
    
    for player in player_tups:
        player, shipyard_x, shipyard_y = player
        if int(player) == my_id:
            frame[shipyard_y, shipyard_x, 3] = 1.
            my_shipyard = shipyard_y, shipyard_x
        else:
            frame[shipyard_y, shipyard_x, 3] = -1.
            enemy_shipyards.append((shipyard_y, shipyard_x))

    # Prime the GPU
    feed_dict = {frames_node: np.zeros((1, 128, 128, 7), dtype=np.float32),
                 my_player_features_node: np.zeros((1, 12), dtype=np.float32),
                 opponent_features_node: np.zeros((1, 3, 3), dtype=np.float32),
                 is_training: False
                }

    for _ in range(1):
        _ = sess.run([moves_node, generate_node], feed_dict=feed_dict)

    # Send name
    print("jstaker7", flush=True)
    #print([x.name for x in local_device_protos if x.device_type == 'GPU'])
    #sys.stdout.flush()

    # TODO: Lots of things I can delete along the way

    while True:
        frame, turns_left, my_player_features, opponent_features, my_ships, has_ship, my_halite, my_dropoffs, enemy_dropoffs = update_frame(frame, num_players, my_id, map_dim, max_turns)
        
        _frame = np.expand_dims(frame.copy(), 0) # Expects batch dim
        
        cost_to_move = np.floor(_frame[0, :, :, 0].copy() * 0.1)
        can_afford_to_move = (_frame[0, :, :, 2].copy() - cost_to_move) >= -1e-13
 
        _frame[:, :, :, 0] =  (_frame[:, :, :, 0])/1000.
        _frame[:, :, :, 2] =  has_ship * ((_frame[:, :, :, 2])/1000.)
        
        # Center
        _frame, shift = center_frames(_frame, include_shift=True)
        
        _frame, _, padding = pad_replay(_frame, include_padding=True)
        
        lxp, rxp, lyp, ryp = padding
        
        feed_dict = {frames_node: _frame, # TODO: Pad, keep track of where the ships are
                     my_player_features_node: my_player_features,
                     opponent_features_node: opponent_features,
                     is_training: False
                    }

        mo, go, ho, bo, wo = sess.run([m_probs, generate_node, h_logits, b_logits, w_logits], feed_dict=feed_dict)

        logging.info(np.squeeze(wo))

        mo = mo[1]
        go = go[1]

        # TODO: Also get the game state to determine the player to use

        go = np.squeeze(go) > 0 # Raw number, 0 is sigmoid()=0.5
        
        mo = mo[0, lyp:-ryp, lxp:-rxp] # reverse pad
        
        #highest_confidence = np.max(mo, -1)
        #highest_confidence_loc = np.argmax(highest_confidence, [0, 1])

        mo = np.roll(mo, -shift[0], axis=0) # reverse center
        mo = np.roll(mo, -shift[1], axis=1) # reverse center
        
        assert mo.shape[0] == mo.shape[1] == map_dim

        commands = []
        
        # Attempt to reduce collisions. This is a heuristic that I'd love to
        # be handled through the learning process.
        is_taken = np.zeros(mo.shape[:2], dtype=np.bool)
        
        already_taken = enemy_shipyards + enemy_dropoffs + my_dropoffs + [my_shipyard]
        
        # Filter out moves when they can't be afforded
        _my_ships = []
        for ship in my_ships:
            id, x, y, h = ship
            a = can_afford_to_move[y, x]
            c = max(mo[y, x])
            _my_ships.append((id, x, y, h, a, c))

        my_ships = sorted(_my_ships, key=lambda x: (x[4], -x[5]))

        constructed = False
        
        for ship in my_ships:
            id, x, y, h, a, _ = ship
            
            #ranked_choices = sorted(list(zip(valid_moves, mo[y, x])), key=lambda x: x[1], reverse=True)
            
            probs = mo[y, x].copy()
            probs = [np.random.uniform(0, i) for i in probs]
            ranked_choices = sorted(list(zip(valid_moves, probs)), key=lambda x: x[1], reverse=True)
            
            for choice in ranked_choices:
                m, _ = choice # don't need probability for now
            
                # TODO: only allow 1 construction per turn?
                if m == 'c' and (my_halite) >= 4000 and (y, x) not in already_taken and not constructed: # TODO: halite on cell and in ship can technically be included
                    move_cmd = "c {}".format(id)
                    commands.append(move_cmd)
                    my_halite -= 4000
                    constructed = True
                    break
                elif m == 'c':
                    continue
                
                if m != 'o' and not a:
                    continue # can't afford the move
                
                move_ix = valid_moves.index(m)
                y_s, x_s = move_shifts[move_ix]
                loc = (y + y_s)%map_dim, (x + x_s)%map_dim

                if not is_taken[loc[0], loc[1]]:
                    move_cmd = "m {} {}".format(id, m)
                    commands.append(move_cmd)
                    is_taken[loc[0], loc[1]] = 1
                    break

            if turns_left < 35:
                for loc in my_dropoffs + [my_shipyard]:
                    is_taken[loc[0], loc[1]] = 0

        if not is_taken[my_shipyard[0], my_shipyard[1]] and go and my_halite >= 1000:
            commands.append("g")
            my_halite -= 1000

        print(" ".join(commands))
        sys.stdout.flush()


