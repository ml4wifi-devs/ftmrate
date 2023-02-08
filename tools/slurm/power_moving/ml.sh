#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ftmrate_internal/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$ML4WIFI_DIR/envs"

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
DELTA=$4
INTERVAL=$5
VELOCITY=$6
START_POS=$7
MEMPOOL_SHIFT=$8

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/power_moving_${MANAGER_NAME}_v${VELOCITY}_d${DELTA}_i${INTERVAL}_s${SEED}.csv"

WARMUP_TIME=5
FUZZ_TIME=1
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="moving" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --delta="$DELTA" --interval="$INTERVAL" --velocity="$VELOCITY" --startPosition="$START_POS" --simulationTime="56" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --measurementsInterval="0.5" --lossModel="$LOSS_MODEL" --seed="$SEED" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"
