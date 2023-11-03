#!/bin/bash

source $1/frames_venv/bin/activate
sudo $1/frames_venv/bin/python3 -u $1/ftmrate_internal/experiments/send_frames.py --count $2
