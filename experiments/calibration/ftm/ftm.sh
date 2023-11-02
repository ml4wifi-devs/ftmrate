#!/bin/bash

for i in {1..100}; do
    sudo iw dev wlp1s0 measurement ftm_request ../conf
done
