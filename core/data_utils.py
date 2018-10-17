import copy
import json
import zstd
import numpy as np


"""
**** GLOBAL VARIABLES ****
These are set to ensure that replay downloads, training, and live bots are all
based on the same assumptions, e.g., the top-ranked player is so-and-so.
"""

# Keep track of the top 10 players and their ranks
PLAYERS = {'shummie': 1,
}

class Game(object):
    """
    Can be used iteratively in a live game, or used to parse replays.
    """
    def __init__(self):
        self.energy_grid = []
        self.moves = []
        self.entities = []
        self.energy = []
        self.deposited = []
        self.meta = None
        self.map_shape = None
    
    
    def load_replay(self, raw_replay: str):
        replay = json.loads(raw_replay)
    
        meta_keys = ['ENGINE_VERSION', 'GAME_CONSTANTS', 'REPLAY_FILE_VERSION',
                     'game_statistics', 'map_generator_seed',
                     'number_of_players', 'players']
                     
        self.meta_data = {key: replay[key] for key in meta_keys}
        #map_shape = np.array(frames).shape[1:]

    def load_production():
        raw_energy_grid = parsed_replay['production_map']['grid']
        energy_grid = []
        for row in raw_energy_grid:
            energy_grid.append([x['energy'] for x in row])
        first_frame = np.array(energy_grid)

        frames = []
        for frame in parsed_replay['full_frames']:
            prev_frame = first_frame if len(frames) == 0 else frames[-1]
            current_frame = prev_frame.copy()
            for c in frame['cells']:
                current_frame[c['y'], c['x']] = c['production']
            frames.append(current_frame)

    def parse_frame(frame: str):
        """
        Useful for both processing downloaded replays or frames of live games used
        by a bot in production.
        """

    def parse_replay(raw: str):
        path = '/Users/Peace/Desktop/replays/ts2018-halite-3-gold-replays_replay-20181016-000040%2B0000-1539647993-48-48'

        with open(path, 'rb') as infile:
            raw_replay = zstd.loads(infile.read()).decode()

        parsed_replay = json.loads(raw_replay)

        meta_data = {key: parsed_replay[key] for key in ['ENGINE_VERSION',
                                                         'GAME_CONSTANTS',
                                                         'REPLAY_FILE_VERSION',
                                                         'game_statistics',
                                                         'map_generator_seed',
                                                         'number_of_players',
                                                         'players']}
        map_shape = np.array(frames).shape[1:]

    def parse_events():
        events = []
        for frame in parsed_replay['full_frames']: # [:3]
            frame_events = parse_events(frame)
            events.append(frame_events)

    def parse_moves(parsed_replay):

        num_players = len(parsed_replay['full_frames'][1]['moves']) #meta_data['number_of_players']

        valid_moves = ['o', 'n', 'e', 's', 'w']
        moves = []
        generate = np.zeros((len(parsed_replay['full_frames']) - 2, num_players), dtype=np.uint8)
        for ix, frame in enumerate(parsed_replay['full_frames'][1:-1]): # No moves on first or last frames
            frame_moves = np.zeros((*map_shape, num_players), dtype=np.uint8)
            for pid in range(num_players):
                if str(pid) not in frame['moves']:
                    continue
                for move in frame['moves'][str(pid)]:

                    if move['type'] == 'm':
                        mid = move['id']
                        ent = frame['entities'][str(pid)][str(mid)]
                        assert move['direction'] in valid_moves
                        frame_moves[ent['y'], ent['x'], pid] = valid_moves.index(move['direction'])
                    else:
                        generate[ix, pid] = 1
            moves.append(frame_moves)

        return np.array(moves), generate


    def parse_entities(parsed_replay):

        num_players = len(parsed_replay['full_frames'][1]['moves']) #meta_data['number_of_players']
        num_features = 2
        entities = []
        for ix, frame in enumerate(parsed_replay['full_frames'][1:-1]): # No enties on first frame (last doesn't matter)
            frame_entities = np.zeros((*map_shape, num_features+num_players), dtype=np.uint8)
            for pid in range(num_players):
                for ent in frame['entities'][str(pid)]:
                    ent = frame['entities'][str(pid)][ent]
                    frame_entities[ent['y'], ent['x'], 0] = ent['energy']
                    frame_entities[ent['y'], ent['x'], 1] = int(ent['is_inspired'])
                    frame_entities[ent['y'], ent['x'], pid+num_features] = 1
            entities.append(frame_entities)

        return np.array(entities)

energy = [[f['energy'][y] for y in sorted(f['energy'].keys())] for f in parsed_replay['full_frames']]
energy = np.array(energy)

player_names = [x['name'].split(' ')[0] for x in meta_data['players']]

    deposited = [x['deposited'] for x in parsed_replay['full_frames']]
deposited = [[x[y] if y in x else 0 for y in ['0', '1', '2', '3']] for x in deposited]

def download_replays():
    """
    To be run on cron. Each night, download the recent replays for the top N
    players.
    """
