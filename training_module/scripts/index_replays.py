try:
    import os
    import gzip
    import pickle

    from core.data_utils import Game

    path = '/Users/Peace/Desktop/replays'

    if not os.path.exists(path):
        path = '/home/staker/Projects/halite/replays'

    index = {}

    # Index the days
    for day in os.listdir(path):
        if day[0] == '.' or day == 'INDEX.pkl':
            continue
        day_dir = os.path.join(path, day)
        
        day_index_path = os.path.join(day_dir, 'INDEX.pkl')
        
        if os.path.exists(day_index_path):
            continue
        
        day_index = {}
        
        for r_name in os.listdir(day_dir):
            if r_name[0] == '.':
                continue
            rp = os.path.join(day_dir, r_name)
            game = Game()
            game.load_replay(rp, meta_only=True)

            day_index[r_name] = game.meta_data

        with gzip.open(day_index_path, 'wb') as outfile:
            pickle.dump(day_index, outfile)


    # Compile master index
    master_index = {}
    for day in os.listdir(path):
        if day[0] == '.' or day == 'INDEX.pkl':
            continue
        day_dir = os.path.join(path, day)
        
        day_index_path = os.path.join(day_dir, 'INDEX.pkl')

        with gzip.open(day_index_path, 'rb') as infile:
            day_index = pickle.load(infile)

            for r_name in day_index:
                master_index[r_name] = day_index[r_name]

    master_index_path = os.path.join(path, 'INDEX.pkl')

    if os.path.exists(master_index_path):
        os.remove(master_index_path)

    with gzip.open(master_index_path, 'wb') as outfile:
        pickle.dump(master_index, outfile)

    print('Indexing complete\n')

except Exception as e:
    print(e)
