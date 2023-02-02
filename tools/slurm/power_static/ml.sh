#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ftmrate_internal/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$ML4WIFI_DIR/envs"
export PYTHONPATH=$HOME/ftmrate_internal/

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
N_WIFI=$4
DISTANCE=$5
DELTA=$6
INTERVAL=$7
SIM_TIME=$8
MEMPOOL_SHIFT=$9

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/power_static_${MANAGER_NAME}_dist${DISTANCE}_n${N_WIFI}_delta${DELTA}_i${INTERVAL}_s${SEED}.csv"

WARMUP_TIME=$(( N_WIFI + 4))
FUZZ_TIME=$(( N_WIFI / 2 + 2 ))
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="distance" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --distance="$DISTANCE" --nWifi="$N_WIFI" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --seed="$SEED" --interval="$INTERVAL" --delta="$DELTA" --adaptPower --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"
