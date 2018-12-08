try:
    import os
    #import zstd
    #import json
    import datetime

    from training_module.hlt_client.download_game import download

    outpath = '/Users/Peace/Desktop/replays'

    if not os.path.exists(outpath):
        outpath = '/home/staker/Projects/halite/replays'

    offset = 1 # To ensure the day is done (timezone differences)

    today = datetime.datetime.now() - datetime.timedelta(days=offset)
    today = str(today).split('-')[:3]
    today[-1] = today[-1].split(' ')[0]
    #today = [str(int(x)) for x in today]

    date = ''.join(today)

#    date = '20181108'
    for date in [date]:

      _outpath = os.path.join(outpath, date)

      os.mkdir(_outpath)

      download(_outpath, date, False)

      print(date)

    #for replay_name in os.listdir(outpath):
    #    path = os.path.join(outpath, replay_name)
    #    with open(path, 'rb') as infile:
    #        raw_replay = zstd.loads(infile.read()).decode()
    #        print(raw_replay)
    #        sdfsf
except Exception as e:
    print(e)
