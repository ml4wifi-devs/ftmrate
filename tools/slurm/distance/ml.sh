#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ftmrate/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

cd "$ML4WIFI_DIR/envs"

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
N_WIFI=$4
DISTANCE=$5
SIM_TIME=$6
MEMPOOL_SHIFT=$7
THRESHOLD=${8:-0}

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

if [[ $THRESHOLD -eq 0 ]]; then
  CSV_PATH="$TOOLS_DIR/outputs/distance_${MANAGER_NAME}_d${DISTANCE}_n${N_WIFI}_s${SEED}.csv"
else
  CSV_PATH="$TOOLS_DIR/outputs/distance_${MANAGER_NAME}_d${DISTANCE}_n${N_WIFI}_s${SEED}_t${THRESHOLD}.csv"
fi

WARMUP_TIME=$(( N_WIFI + 4))
FUZZ_TIME=$(( N_WIFI / 2 + 2 ))
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="distance" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --distance="$DISTANCE" --nWifi="$N_WIFI" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --seed="$SEED" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY" --threshold="$THRESHOLD"
