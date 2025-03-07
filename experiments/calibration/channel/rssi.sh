#!/bin/bash

sudo timeout 10 tcpdump -i mon0 -e 'ether host 00:c2:c6:e6:9a:ec' -s 70
