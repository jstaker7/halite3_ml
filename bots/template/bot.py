#!/usr/bin/env python3

import os
import sys
import time
import logging

logging.basicConfig(filename='./test.log',
                            #filemode='a',
                            #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            #datefmt='%H:%M:%S',
                            level=logging.DEBUG)

#logger = logging.getLogger('./test.log')
#logger.setLevel(30)
logging.info('test')

import numpy as np

# Turning off logging doesn't seem to be working
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.WARN) # Turn this on before uploading
#tf.logging.set_verbosity(0)
#tf.logging.set_verbosity()

import hlt

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

def update_frame(frame):

    turn_number = input()
    #logging.info(turn_number)
    
    num_players = 2
    
    for _ in range(num_players):
        player, num_ships, num_dropoffs, halite = map(int, input().split())
        logging.info(player)
        logging.info(num_ships)
        logging.info(num_dropoffs)
        logging.info(halite)
    
        ships = {id: (id, x, y, h) for (id, x, y, h) in [map(int, input().split()) for _ in range(num_ships)]}
        dropoffs = {id: (id, x, y) for (id, x, y) in [map(int, input().split()) for _ in range(num_dropoffs)]}
        
    for _ in range(int(input())):
        x, y, h = map(int, input().split())
        frame[y, x, 0] = h

    return frame

def initialize_frame():
    raw_constants = input()
    
    #logging.info(raw_constants)

    #json.loads(raw_constants)

    num_players, my_id = map(int, input().split())
    
    #logging.info(num_players)
    #logging.info(my_id)
    
    num_players = 2
    
    for player in range(num_players):
        player, shipyard_x, shipyard_y = map(int, input().split())
        #logging.info(player)
        #logging.info(shipyard_x)
        #logging.info(shipyard_y)
    
    map_width, map_height = map(int, input().split())
    #game_map = [[None for _ in range(map_width)] for _ in range(map_height)]
    game_map = []
    for _ in range(map_height):
        row = [int(x) for x in input().split()]
        game_map.append(row)

    frame = np.array(game_map)
    #logging.info(frame)

    # Not all info has been provided yet. We need to call an update to fill in
    # the missing info.

    return frame

# Load the model
with tf.Session() as sess:
    tf.train.import_meta_graph(os.path.join(cd, "model.ckpt.meta"))
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(cd, "model.ckpt"))

    frames_node = tf.get_collection('frames')[0]
    can_afford_node = tf.get_collection('can_afford')[0]
    turns_left_node = tf.get_collection('turns_left')[0]
    my_ships_node = tf.get_collection('my_ships')[0]
    moves_node = tf.get_collection('m_logits')[0]
    generate_node = tf.get_collection('g_logits')[0]
    
    frame = initialize_frame()

    # Send name
    print("jstaker7", flush=True)
    #sys.stdout.flush()

#    while True:
#        raw_input = input()
#        logging.info(raw_input)
#        time.sleep(1)

    while True:
        frame = update_frame(frame)
        # You extract player metadata and the updated map metadata here for convenience.
        #me = game.me
        #logging.info(me)
        #game_map = game.game_map
        #logging.info(game_map)
        #sfsdfdfs
        # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
        #   end of the turn.
        command_queue = []

        for ship in me.get_ships():
            # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
            #   Else, collect halite.
            if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
                command_queue.append(
                    ship.move(
                        random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
            else:
                command_queue.append(ship.stay_still())

        # If the game is in the first 200 turns and you have enough halite, spawn a ship.
        # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
        if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            command_queue.append(me.shipyard.spawn())

        # Send your moves back to the game environment, ending this turn.
        game.end_turn(command_queue)


