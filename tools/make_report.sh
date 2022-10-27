#!/bin/bash

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"
ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/tools}"

OUTPUT_FILE="$HOME/report.md"

printf "# Commits to reproduce results\n\n" > "$OUTPUT_FILE"

printf "## ns-3-dev\n" >> "$OUTPUT_FILE"
cd "$NS3_DIR"
git log | head -n 6 >> "$OUTPUT_FILE"

printf "## ml4wifi\n" >> "$OUTPUT_FILE"
cd "$ML4WIFI_DIR"
git log | head -n 6 >> "$OUTPUT_FILE"

printf "## tools\n" >> "$OUTPUT_FILE"
cd "$TOOLS_DIR"
git log | head -n 6 >> "$OUTPUT_FILE"

printf "# Command to reproduce results\n\n\n\n" >> "$OUTPUT_FILE"

printf "# Comments\n\n" >> "$OUTPUT_FILE"
