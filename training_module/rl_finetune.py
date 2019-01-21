import os
import time
import json
import multiprocessing
import subprocess
from random import shuffle

import numpy as np

import tensorflow as tf

import shutil

num_per_update = 2
map_sizes = [32, 40, 48, 56, 64]
num_players = [2, 4]

if os.path.exists('/Users/Peace'):
    bot_cmd = '"python /Users/Peace/Projects/halite3_ml/bots/rl1/MyBot.py"'
    engine = '/Users/Peace/Desktop/tourney/halite'
    outdir = '/Users/Peace/Projects/halite_games'
    save_template = '/Users/Peace/Projects/halite3_ml/bots/rl_model/model_{}.ckpt'
else:
    bot_cmd = '"python /home/staker/Projects/halite/rl_stuff/rl1/MyBot.py"'
    engine = '/home/staker/Projects/halite/rl_stuff/halite'
    outdir = '/home/staker/Projects/halite/rl_stuff/halite_games'
    save_template = '/home/staker/Projects/halite/rl_stuff/rl_model/model_{}.ckpt'

template = '{e} --no-timeout --no-logs --no-replay --results-as-json --width {m} --height {m} {p} -s {s} > {d}/{s}.json'

def run_game(_):
    np.random.seed(None)
    seed = np.random.randint(303946417)

    size = np.random.choice(map_sizes)
    n = np.random.choice(num_players)

    players = ' '.join([bot_cmd]*n)

    cmd = template.format(**{'e': engine,
                             'm': size,
                             'p': players,
                             's': seed,
                             'd': outdir
                            })


    #return
    
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

with tf.variable_scope("reinforcement"):
    state_node = tf.placeholder(tf.float32, [None, 64])
    action_node = tf.placeholder(tf.uint8, [None, 1])
    did_win_node = tf.placeholder(tf.float32, [None, 1])
    is_training = tf.placeholder(tf.bool)
    tf.add_to_collection('rl_is_training', is_training)
    tf.add_to_collection('rl_state', state_node)

    rl_l1 = tf.layers.dense(state_node, 32, activation=tf.nn.relu, name='rl1')
    rl_l1 = tf.layers.batch_normalization(rl_l1, training=is_training, name='rlbn1')
    rl_l2 = tf.layers.dense(rl_l1, 32, activation=tf.nn.relu, name='rl2')
    rl_l2 = tf.layers.batch_normalization(rl_l2, training=is_training, name='rlbn2')

    num_production_players = 2
    player_probs = tf.layers.dense(rl_l2, num_production_players, activation=None, name='pl_probs')

    tf.add_to_collection('player_probs', tf.nn.softmax(player_probs))
    
#    log_prob = tf.log(tf.nn.softmax(player_probs))
#    loss = -tf.reduce_sum(tf.mul(log_prob, did_win))

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=action_node,
                                            logits=player_probs,
                                            dim=-1)

    #loss = tf.reduce_mean(losses * did_win_node)
    loss = tf.reduce_mean(losses)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

if __name__ == '__main__':
    #count = multiprocessing.cpu_count()//3
    count = 1
    pool = multiprocessing.Pool(processes=count)
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        shutil.rmtree(outdir)
        os.mkdir(outdir)

    batch_size = 256
    with tf.Session() as sess:
        saver = tf.train.Saver()
    
        # Initialize the model
        tf.initializers.global_variables().run()
        saver.save(sess, save_template.format('recent'))
    
        for opt_step in range(99999999):
            print(opt_step)
            pool.map(run_game, range(num_per_update))
            
            time.sleep(3) # Time to clear data to disk

            results = []
            for item in os.listdir(outdir):
                if '.json' in item:
                    with open(os.path.join(outdir, item), 'r') as infile:
                        players = json.load(infile)['stats']
                        map_id = item.split('.')[0]
                        for pid in players:
                            did_win = 1 == players[pid]['rank']
                            
                            if not did_win:
                                continue
                            
                            a_path = os.path.join(outdir, '{}_{}_a.npy'.format(pid, map_id))
                            s_path = os.path.join(outdir, '{}_{}_s.npy'.format(pid, map_id))
                            
                            if not os.path.exists(a_path) or not os.path.exists(s_path):
                                continue
                            
                            actions = np.load(a_path)
                            states = np.load(s_path)
                            # did_win = 1 if 1 == players[pid]['rank'] else -1
                            #results.append((actions, states, np.ones(actions.shape)*did_win))
                            for a, s in zip(actions, states):
                                results.append((a, s, did_win))

            shuffle(results)

            actions, states, adv, = zip(*results)

            actions = np.array(actions)
            states = np.squeeze(np.array(states))
            adv = np.array(adv)

            for i in range(0, max(actions.shape[0]//batch_size, 1), batch_size):
                a_batch = np.expand_dims(actions[i:i+batch_size], -1)
                s_batch = states[i:i+batch_size]
                v_batch = np.expand_dims(adv[i:i+batch_size], -1)
                
                feed_dict = {state_node: s_batch,
                             action_node: a_batch,
                             did_win_node: v_batch,
                             is_training: True
                            }

                sess.run(optimizer, feed_dict=feed_dict)

            saver.save(sess, save_template.format(opt_step))
            saver.save(sess, save_template.format('recent'))

            shutil.rmtree(outdir)
            os.mkdir(outdir)

            # TODO: place bots on GPU
