#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ftmrate_internal/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$ML4WIFI_DIR/envs"

SEED_SHIFT=$1
MANAGER=$7
MANAGER_NAME=$8
N_WIFI=$4
DISTANCE=$2
DELTA=$3
INTERVAL=$5
SIM_TIME=$6
MEMPOOL_SHIFT=$9

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/${MANAGER_NAME}_adopt_nwifi${N_WIFI}_dist${DISTANCE}_delta${DELTA}_interval${INTERVAL}_seed${SEED}.csv"

WARMUP_TIME=$(( N_WIFI + 4))
FUZZ_TIME=$(( N_WIFI / 2 + 2 ))
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="distance" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --distance="$DISTANCE" --nWifi="$N_WIFI" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --seed="$SEED" --interval="$INTERVAL" --delta="$DELTA" --adaptPower --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"
