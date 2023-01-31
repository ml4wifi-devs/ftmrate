#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3-dev}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate_internal/tools}"

cd "$NS3_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR/build/lib
cd build/scratch

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
N_WIFI=$4
DISTANCE=$5
DELTA=$6
INTERVAL=$7
SIM_TIME=$8

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$VAR_POWER_DIR/data/${MANAGER_NAME}_adopt_nwifi${N_WIFI}_dist${DISTANCE}_delta${DELTA}_interval${INTERVAL}_seed${SEED}"

WARMUP_TIME=$(( N_WIFI + 4))
FUZZ_TIME=$(( N_WIFI / 2 + 2 ))
LOSS_MODEL="Nakagami"

./ns3.36.1-stations-optimized --manager="$MANAGER" --managerName="$MANAGER_NAME" --distance="$DISTANCE" --nWifi="$N_WIFI" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --fuzzTime="$FUZZ_TIME" --lossModel="$LOSS_MODEL" --RngRun="$SEED" --interval="$INTERVAL" --delta="$DELTA" --csvPath="$CSV_PATH"
