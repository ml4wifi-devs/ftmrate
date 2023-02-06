#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$NS3_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR/build/lib
cd build/scratch

SEED=$1
MODE=$2
SNR=$3

CSV_PATH="$TOOLS_DIR/outputs/success-probability/success-probability_m${MODE}_snr${SNR}_s${SEED}.csv"

SIM_TIME=1000
WARMUP_TIME=10
FUZZ_TIME=5
LOSS_MODEL="Nakagami"

./ns3.36.1-success-probability-optimized --mode="$MODE" --snr="$SNR" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --RngRun="$SEED" --csvPath="$CSV_PATH"
