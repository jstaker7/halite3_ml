#!/bin/sh

source activate cdeep3

export PYTHONPATH=/home/staker/Projects/halite/halite3_ml

python /home/staker/Projects/halite/halite3_ml/training_module/scripts/download_replays.py >> /home/staker/logs/download_script.log

source deactivate
