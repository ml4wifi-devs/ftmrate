#!/bin/bash

source $1/ftmrate_venv/bin/activate
sudo $1/ftmrate_venv/bin/python3.8 -u $1/ftmrate_internal/experiments/ftmrate.py
