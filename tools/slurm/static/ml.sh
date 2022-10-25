#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/tools}"

cd "$ML4WIFI_DIR/ml4wifi/envs"

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
N_WIFI=$4
DISTANCE=$5
SIM_TIME=$6
MEMPOOL_SHIFT=$7

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/static_${MANAGER_NAME}_d${DISTANCE}_n${N_WIFI}_s${SEED}.csv"

WARMUP_TIME=$(( N_WIFI + 4))
FUZZ_TIME=$(( N_WIFI / 2 + 2 ))
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="static" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --distance="$DISTANCE" --nWifi="$N_WIFI" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --seed="$SEED" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"