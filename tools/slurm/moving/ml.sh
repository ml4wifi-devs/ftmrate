#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

ML4WIFI_DIR="${ML4WIFI_DIR:=$HOME/ftmrate/ml4wifi}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

cd "$ML4WIFI_DIR/envs"

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
VELOCITY=$4
SIM_TIME=$5
INTERVAL=$6
MEMPOOL_SHIFT=$7
WALL_INTERVAL=${8:-0}
WALL_LOSS=${9:-0}

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/moving_${MANAGER_NAME}_v${VELOCITY}_s${SEED}.csv"

WARMUP_TIME=5
FUZZ_TIME=1
LOSS_MODEL="Nakagami"

python3 ml_wifi_manager.py --scenario="moving" --ml_manager="$MANAGER" --managerName="$MANAGER_NAME" --velocity="$VELOCITY" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --measurementsInterval="$INTERVAL" --lossModel="$LOSS_MODEL" --seed="$SEED" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY" --wallInterval="$WALL_INTERVAL" --wallLoss="$WALL_LOSS"
