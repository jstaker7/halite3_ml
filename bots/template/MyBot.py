#!/usr/bin/env python3

import os
import sys
import time
import logging
import json

#logger = logging.getLogger('./test.log')
#logger.setLevel(30)
#logging.info('test')

import numpy as np

# Turning off logging doesn't seem to be working
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.WARN) # Turn this on before uploading
#tf.logging.set_verbosity(0)
#tf.logging.set_verbosity()

cd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cd)
from data_utils import Game

#def read_input():
#    """
#    Reads input from stdin, shutting down logging and exiting if an EOFError occurs
#    :return: input read
#    """
#    try:
#        return input()
#    except EOFError as eof:
#        logging.shutdown()
#        raise SystemExit(eof)
#
## Rename for ease of use
#input = read_input


#for i in range(5):
#    raw_input = input()
#    logging.info(raw_input)
#    time.sleep(1)
#ghvhgv

# Required inputs:
# production, has_ship, entity_energies, factories, has_dropoff
# can_afford
# turns_left

def update_frame(frame, num_players, my_id):

    # Some channels get refreshed entirely
    frame[:, :, [1,2,4]] = 0

    turn_number = input()
    
    # TODO: Add in feature to account for my halite vs the opponents
    
    my_halite = None
    my_ships = None
    
    has_ship = np.zeros((frame.shape[1], frame.shape[1]), dtype=np.uint8)

    for _ in range(num_players):
        player, num_ships, num_dropoffs, halite = [int(x) for x in input().split()]
        
        ships = [[int(x) for x in input().split()] for _ in range(num_ships)]
        dropoffs = [[int(x) for x in input().split()] for _ in range(num_dropoffs)]
        
        if player == my_id:
            my_halite = halite
            my_ships = ships
        
        for ship in ships:
            id, x, y, h = ship
            frame[y, x, 2] = h
            if player == my_id:
                frame[y, x, 1] = 1.
            else:
                frame[y, x, 1] = -1.
            
            has_ship[y, x] = 1
    
        for dropoff in dropoffs:
            id, x, y = dropoff
            if player == my_id:
                frame[y, x, 4] = 1.
            else:
                frame[y, x, 4] = -1.

    
    for _ in range(int(input())):
        x, y, h = [int(x) for x in input().split()]
        frame[y, x, 0] = h

    assert my_halite is not None

    can_afford_both = my_halite > 4999.
    can_afford_drop = my_halite > 3999.
    can_afford_ship = my_halite > 999.

    can_afford = np.stack([can_afford_ship, can_afford_drop, can_afford_both], -1)
    
    has_ship = np.expand_dims(has_ship, 0)

    return frame, turn_number, can_afford, my_ships, has_ship

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

game = Game()

valid_moves = ['o', 'n', 'e', 's', 'w', 'c']

# Load the model
with tf.Session() as sess:
    tf.train.import_meta_graph(os.path.join(cd, "model.ckpt.meta"))
    saver = tf.train.Saver()
    #saver.restore(sess, os.path.join(cd, "model.ckpt"))
    saver.restore(sess, os.path.join(cd, "model.ckpt"))
    

    frames_node = tf.get_collection('frames')[0]
    can_afford_node = tf.get_collection('can_afford')[0]
    turns_left_node = tf.get_collection('turns_left')[0]
    moves_node = tf.get_collection('m_logits')[0]
    generate_node = tf.get_collection('g_logits')[0]
    
    max_turns, num_players, my_id, halite, player_tups, map_dim = get_initial_data()
    
    logging.basicConfig(filename='{}-bot.log'.format(my_id),
                                filemode='w',
                                #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                #datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    
    frame = np.zeros((map_dim, map_dim, 5), dtype=np.float32)
    frame[:, :, 0] = halite
    
    for player in player_tups:
        player, shipyard_x, shipyard_y = player
        if int(player) == my_id:
            frame[shipyard_y, shipyard_x, 3] = 1.
        else:
            frame[shipyard_y, shipyard_x, 3] = -1.
    
    # Send name
    print("jstaker7", flush=True)
    #sys.stdout.flush()

#    while True:
#        raw_input = input()
#        logging.info(raw_input)
#        time.sleep(1)

    while True:
        # TODO: Still need to center the frame according to my shipyard
        # And then shift it back again?
        frame, turn_number, can_afford, my_ships, has_ship = update_frame(frame, num_players, my_id)
        
        _frame = np.expand_dims(frame.copy(), 0) # Expects batch dim
        
        _frame[:, :, :, 0] =  (_frame[:, :, :, 0] - 500.)/500.
        _frame[:, :, :, 2] =  has_ship * ((_frame[:, :, :, 2] - 500.)/500.)
        
        # Center
        _frame, shift = game.center_frames(_frame, include_shift=True)
        
        turns_left = (float(max_turns) - float(turn_number))/200. - 1. # TODO: Is this off by one?
        turns_left = np.expand_dims(turns_left, 0)
        
        _frame, _, padding = game.pad_replay(_frame, include_padding=True)
        
        lxp, rxp, lyp, ryp = padding
        
        logging.info(_frame.shape)
        logging.info(_frame[0, 124:-124, 124:-124, :-1])
        
        feed_dict = {frames_node: _frame, # TODO: Pad, keep track of where the ships are
                     can_afford_node: np.expand_dims(can_afford, 0),
                     turns_left_node: np.expand_dims(turns_left, 0)
                    }

        mo, go = sess.run([moves_node, generate_node], feed_dict=feed_dict)

        go = np.squeeze(go) > 0 # Raw number, 0 is sigmoid()=0.5
        
        mo = np.argmax(mo, -1)
        logging.info(np.max(mo))
        # Remove batch dim; reverse transformations
        mo = mo[0, lyp:-ryp, lxp:-rxp] # reverse pad
        #logging.info(mo[10:30, 10:30])
        mo = np.roll(mo, -shift[0], axis=0) # reverse center
        mo = np.roll(mo, -shift[1], axis=1) # reverse center
        
        assert mo.shape[0] == mo.shape[1] == map_dim


        
        logging.info(np.max(mo))
        logging.info(mo[10:30, 10:30])

        #logging.info(mo)

        commands = []

        if go:
            commands.append("g")
        logging.info('hmm')
        logging.info(len(my_ships))
        for ship in my_ships:
            id, x, y, h = ship
            logging.info((id, x, y))
            move_ix = mo[y, x]
            move = valid_moves[move_ix]
            if move == 'c':
                move_cmd = "c {}".format(id)
            else:
                move_cmd = "m {} {}".format(id, move)
                
            commands.append(move_cmd)


        print(" ".join(commands))
        sys.stdout.flush()


