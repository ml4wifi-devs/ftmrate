#!/bin/bash

TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"
OUTPUT_FILE="all_results.csv"

cd "$TOOLS_DIR/outputs"
echo "mobility,manager,delta,interval,velocity,distance,time,nWifi,nWifiReal,seed,throughput" > "$OUTPUT_FILE"

for file in *.csv; do
    if [[ "$file" != "$OUTPUT_FILE" ]]; then
        cat "$file" >> "$OUTPUT_FILE"
    fi
done
