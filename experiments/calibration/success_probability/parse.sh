#!/bin/bash

FILENAME="data_raw.csv"
echo "mode,distance,k,n" > $FILENAME

N=100

for file in *.pcap; do
    echo $file

    K=`tshark -Y 'wlan.ta == 00:c2:c6:e6:9a:ec && frame.len > 1000' -r $file | wc -l`
    K=`echo $K | sed 's/ *$//g'`

    MCS=$(cut -d '_' -f1 <<< "$file")
    MCS=$(cut -d 's' -f2 <<< "$MCS")

    D=$(cut -d '_' -f2 <<< "$file")
    D=$(cut -d 'd' -f2 <<< "$D")

    echo "$MCS,$D,$K,$N" >> "$FILENAME"
done
