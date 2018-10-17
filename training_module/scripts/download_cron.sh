#!/bin/bash

export PYTHONPATH=/home/staker/Projects/halite/halite3_ml

source activate cdeep3

/home/staker/miniconda3/envs/cdeep3/bin/python3.6 /home/staker/Projects/halite/halite3_ml/training_module/scripts/download_replays.py >> /home/staker/logs/download_script.log

source deactivate
