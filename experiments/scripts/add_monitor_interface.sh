#!/bin/bash

sudo iw phy phy0 interface add mon0 type monitor
sudo iw dev
sudo ifconfig mon0 up
