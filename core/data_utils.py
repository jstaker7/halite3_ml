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
    
    'Load' methods are used to populate values from a relay.
    'Parse' methods are used on individual frames (used for both replays and iteratively)
    """
    def __init__(self):
        self.production = []
        self.moves = []
        self.generate = []
        self.entities = []
        self.energy = []
        self.deposited = []
        self.meta = None
        self.map_shape = None
        self.factories = []
        self.dropoffs = []
        self.events = []
        
        self.replay = None # Store the whole replay for easier debugging
    
    def get_replay(self):
        return self.replay
    
    def parse_events(self, frame):
        frame_events = []
        for e in frame['events']:
            if e['type'] == 'spawn':
                event = [0, e['location']['x'], e['location']['y'], e['id'], e['owner_id']]
            elif e['type'] == 'construct':
                event = [1, e['location']['x'], e['location']['y'], e['id'], e['owner_id']]
            elif e['type'] == 'shipwreck':
                event = [2, e['location']['x'], e['location']['y']] + [x for x in e['ships']]
            else:
                assert False, print(e)
            
            frame_events.append(event)
        return frame_events
    
    def parse_factories(self):
        players = self.meta_data['players']
        factories = []
        assert list(range(len(players))) == [x['player_id'] for x in players], print(players)
        for i in range(len(players)):
            f = players[i]['factory_location']
            factories.append([f['x'], f['y']])
        
        return np.array(factories)
    
    def load_dropoffs(self):
        dropoffs = []
        for f in self.events:
            prev = [] if len(dropoffs) == 0 else dropoffs[-1]
            frame_dropoffs = [] + prev # Dropoffs seem to be permanent
            for e in f:
                if e[0] == 1:
                    frame_dropoffs.append([e[1], e[2], e[4]])
            dropoffs.append(frame_dropoffs)
        return dropoffs
    
    
    def load_replay(self, path: str, meta_only=False):
        with open(path, 'rb') as infile:
            raw_replay = zstd.loads(infile.read()).decode()
        replay = json.loads(raw_replay)
        
        self.replay = replay
    
        meta_keys = ['ENGINE_VERSION', 'GAME_CONSTANTS', 'REPLAY_FILE_VERSION',
                     'game_statistics', 'map_generator_seed',
                     'number_of_players', 'players']
                     
        self.meta_data = {key: replay[key] for key in meta_keys}
        
        if meta_only:
            return
        
        self.events = [self.parse_events(f) for f in replay['full_frames']]
        self.factories = self.parse_factories()
        
        self.production = self.load_production(replay)
        self.moves, self.generate = self.load_moves(replay)
        self.entities = self.load_entities(replay)
        self.energy = self.load_energy(replay)
        self.deposited = self.load_deposited(replay)
        self.dropoffs = self.load_dropoffs()
        
        # player_names = [x['name'].split(' ')[0] for x in meta_data['players']]
        
        # Some of these need to be trimmed
        
        # First is just an init frame
        self.events = self.events[1:]
        
        # Last reflects what the production will be if moves were made on last frame
        # (but there aren't any on last). Also, we don't care what the production looks like
        # on the last frame (because we will have already made our last move).
        # The indexing is weird because the replays show what production would be after moves
        # are made.
        self.production = self.production[:-2]
        self.dropoffs = self.dropoffs[:-2]
        
        # As if moved after last frame, but there are no moves
        self.energy = self.energy[:-1]
        self.deposited = self.deposited[:-1]

    def load_production(self, replay):
        pm = replay['production_map']
        assert list(pm['grid'][0][0].keys()) == ['energy']
        raw_energy_grid = pm['grid']
        energy_grid = []
        for row in raw_energy_grid:
            energy_grid.append([x['energy'] for x in row])
        first_frame = np.array(energy_grid)

        production = []
        for frame in replay['full_frames']:
            prev_frame = first_frame if len(production) == 0 else production[-1]
            current_frame = prev_frame.copy()
            for c in frame['cells']:
                current_frame[c['y'], c['x']] = c['production']
            production.append(current_frame)
        return np.array(production)

    def load_moves(self, replay):
        map_size = self.meta_data['GAME_CONSTANTS']['DEFAULT_MAP_HEIGHT'] # Assuming square
        num_players = len(replay['full_frames'][1]['moves'])

        valid_moves = ['o', 'n', 'e', 's', 'w', 'c']
        moves = []
        generate = np.zeros((len(replay['full_frames']) - 2, num_players), dtype=np.uint8)
        for ix, frame in enumerate(replay['full_frames'][1:-1]): # No moves on first or last frames
            frame_moves = np.zeros((map_size, map_size, num_players), dtype=np.uint8)
            for pid in range(num_players):
                if str(pid) not in frame['moves']:
                    continue
                for move in frame['moves'][str(pid)]:

                    if move['type'] == 'm':
                        mid = move['id']
                        ent = frame['entities'][str(pid)][str(mid)]
                        assert move['direction'] in valid_moves
                        frame_moves[ent['y'], ent['x'], pid] = valid_moves.index(move['direction'])
                    elif move['type'] == 'g':
                        generate[ix, pid] = 1
                    elif move['type'] == 'c':
                        mid = move['id']
                        ent = frame['entities'][str(pid)][str(mid)]
                        frame_moves[ent['y'], ent['x'], pid] = valid_moves.index('c')
                    else:
                        assert False, print(move)
            moves.append(frame_moves)

        return np.array(moves), generate


    def load_entities(self, replay):
        
        map_size = self.meta_data['GAME_CONSTANTS']['DEFAULT_MAP_HEIGHT'] # Assuming square
        num_players = len(replay['full_frames'][1]['moves']) #meta_data['number_of_players']
        num_features = 2
        entities = []
        for ix, frame in enumerate(replay['full_frames'][1:-1]): # No enties on first frame (last doesn't matter)
            frame_entities = np.zeros((map_size, map_size, num_features+num_players), dtype=np.int32)
            for pid in range(num_players):
                for ent in frame['entities'][str(pid)]:
                    ent = frame['entities'][str(pid)][ent]
                    frame_entities[ent['y'], ent['x'], 0] = ent['energy']
                    frame_entities[ent['y'], ent['x'], 1] = int(ent['is_inspired'])
                    frame_entities[ent['y'], ent['x'], pid+num_features] = 1
            entities.append(frame_entities)

        return np.array(entities)
    
    def load_energy(self, replay):
        energy = [[f['energy'][y] for y in sorted(f['energy'].keys())] for f in replay['full_frames']]
        return np.array(energy)

    def load_deposited(self, replay):
        deposited = [x['deposited'] for x in replay['full_frames']]
        deposited = [[x[y] if y in x else 0 for y in ['0', '1', '2', '3']] for x in deposited]
        return np.array(deposited)

    def get_training_frames(self, pid=None, pname=None, v=None):
        # pid OR name. Bot version is option (only when name given)
        # if all None, take random player

        if pid is None and pname is not None:
            for p in self.meta_data['players']:
                if pname == p['name'].split(' ')[0]:
                    pid = p['player_id']
                    break
                    
        assert pid is not None, "Error: player not found in replay"
        
        map_shape = self.production.shape[1], self.production.shape[2]
        factories = np.zeros((*map_shape, 1), dtype=np.float32)
        for fx, fy in self.factories:
            factories[fy, fx] = -1 # Assume all are enemies

        # Then set for the player of interest
        fx, fy = self.factories[pid]
        factories[fy, fx] = 1

        # normalize some of the arrays
        production = (self.production - 500.)/1000. # Guessing on norm values for now

        entities = self.entities.copy()
        has_ship = entities > 0
        has_ship_mask = np.sum(has_ship.astype(np.float32), -1)
        my_ships = has_ship[:, :, :, pid].copy()

        entities = (np.sum(entities, -1) - 500.)/1000.


        has_ship = -1 * np.sum(has_ship.astype(np.float32), axis=-1)
        #print(entities.shape)
        #print(my_ships.shape)
        has_ship[my_ships] = 1
        entities *= has_ship_mask
        #print(has_ship[100, 20:40, 20:40])
        #print(entities[100, 20:40, 20:40])
        has_ship = np.expand_dims(has_ship, -1)
        entities = np.expand_dims(entities, -1)

        dropoffs = self.dropoffs
        has_dropoff = np.zeros((production.shape[0], *map_shape, 1), dtype=np.float32)
        assert len(dropoffs) == len(production)
        for ix, ds in enumerate(dropoffs):
            for d in ds:
                x, y, oid = d
                v = 1. if oid == pid else -1.
                has_dropoff[ix, y, x] = v

        # factories need to be duplicated accross frames
        factories = np.repeat(np.expand_dims(factories, 0), production.shape[0], 0)

        production = np.expand_dims(production, -1)

        frames = np.concatenate([production, has_ship, entities, factories, has_dropoff], axis=-1)

        # Note: 'no-moves' do not need explicit assignment since 0 means 'still'
        # and the arrays are initialled with zero.
        
        moves = self.moves[:, :, :, pid]

        frames, moves = self.center_frames(frames, moves)

        frames = self.pad_replay(frames)
        
        generate = self.generate[:, pid]
        energy = self.energy[:-1, pid] # -1 because I don't need final state here
        
        can_afford_both = energy > 4999.
        can_afford_drop = energy > 3999.
        can_afford_ship = energy > 999.
        
        can_afford = np.stack([can_afford_ship, can_afford_drop, can_afford_both], -1)
        
        turns_left = np.array(list(range(can_afford.shape[0]-1, -1, -1)))
        turns_left = turns_left/200. - 1.
        
        # Should also add diff between my score and others

        return frames, moves, generate, can_afford, turns_left

    def center_frames(self, frames, moves=None):
        my_factory = frames[0, :, :, 3] > 0
        pos = np.squeeze(np.where(my_factory>0))
        expected_pos = np.squeeze(my_factory.shape)//2 # Assuming always square
        shift = expected_pos - pos
        frames = np.roll(frames, shift[0], axis=1)
        frames = np.roll(frames, shift[1], axis=2)
        
        if moves is not None:
            moves = np.roll(moves, shift[0], axis=1)
            moves = np.roll(moves, shift[1], axis=2)
            return frames, moves
        
        return frames

    def pad_replay(self, frames):
        # Let's do full padding (by reflection)
        frames = np.concatenate([frames, frames, frames], axis=1)
        frames = np.concatenate([frames, frames, frames], axis=2)
        
        # Ensure all get padded to the same max dim
        if frames.shape[1] != 192:
            pad_y1 = (192 - frames.shape[1])//2
            pad_y2 = (192 - frames.shape[1]) - pad_y1
            frames = np.concatenate([frames[:, -pad_y1:], frames, frames[:, :pad_y2]], axis=1)
            
            pad_x1 = (192 - frames.shape[2])//2
            pad_x2 = (192 - frames.shape[2]) - pad_x1
            frames = np.concatenate([frames[:, :, -pad_x1:], frames, frames[:, :, :pad_x2]], axis=2)
        
        return frames
    
    def reverse_padding(self, true_size):
        pass

    def aug(move, frame, iter, frames_padding):
        """
        Augment the data using rotations and mirroring.
        """
        should_mirror = iter>3
        
        frames_padding = np.array([[frames_padding[0][0], frames_padding[0][1]],
                                    [frames_padding[1][0], frames_padding[1][1]]])
        

        num_rotate = iter%4
        
        if should_mirror:
            frame = np.flip(frame, 0)
            move = np.flip(move, 0)
        
            frames_padding = np.array([[frames_padding[0][1], frames_padding[0][0]],
                                        [frames_padding[1][0], frames_padding[1][1]]])
        
        frame = np.rot90(frame, num_rotate)
        move = np.rot90(move, num_rotate)
        
        for _ in range(num_rotate):
            frames_padding = np.array([[frames_padding[1][1], frames_padding[1][0]],
                                       [frames_padding[0][0], frames_padding[0][1]]])
        
        # We need to change the move values to match the augmentation
        # STILL will not change; this is simply reordering the axis
        # Current axis: N:1, E:2, S:3, W:4
        if should_mirror:
            move = move[:, :, (0, 3, 2, 1, 4)]

        shift = np.roll(move[:, :, 1:].copy(), -1*num_rotate, 2)
        move[:, :, 1:] = shift

        return move, frame, frames_padding

