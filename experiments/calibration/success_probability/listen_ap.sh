#!/bin/bash

sudo tcpdump -i mon0  -e 'ether host 00:c2:c6:e6:9a:ec' -w $1
