#!/bin/bash

TOOLS_DIR="${TOOLS_DIR:=$HOME/tools}"
OUTPUT_FILE="success_probability.csv"

cd "$TOOLS_DIR/outputs/success-probability"
echo "mode,snr,n,k" > "$OUTPUT_FILE"

for file in *.csv; do
    if [[ "$file" != "$OUTPUT_FILE" ]]; then
        cat "$file" >> "$OUTPUT_FILE"
    fi
done
