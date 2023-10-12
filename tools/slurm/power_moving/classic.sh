#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$NS3_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR/build/lib
cd build/scratch

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
DELTA=$4
INTERVAL=$5
VELOCITY=$6
START_POS=$7

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/power_moving_${MANAGER_NAME}_v${VELOCITY}_d${DELTA}_i${INTERVAL}_s${SEED}.csv"

WARMUP_TIME=5
FUZZ_TIME=1
LOSS_MODEL="Nakagami"

./moving --manager="$MANAGER" --managerName="$MANAGER_NAME" --delta="$DELTA" --interval="$INTERVAL" --velocity="$VELOCITY" --startPosition="$START_POS" --simulationTime="56" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --measurementsInterval="0.5" --lossModel="$LOSS_MODEL" --RngRun="$SEED" --csvPath="$CSV_PATH"
