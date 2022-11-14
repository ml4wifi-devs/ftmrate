#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
TASKS_PER_NODE=5

MANAGERS=("pf" "kf" "lt")
MANAGERS_NAMES=("PF" "KF" "LT")
MANAGERS_LEN=${#MANAGERS[@]}

SHIFT=0
BASE_SEED=100
BASE_MEMPOOL=3000

run_static() {
  N_REP=10
  N_POINTS=13
  DISTANCE=$1

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < N_POINTS; j++)); do
      N_WIFI=$(( j == 0 ? 1 : 4 * j))
      SIM_TIME=$(( 10 * N_WIFI + 50 ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      SEED_SHIFT=$(( SHIFT + BASE_SEED ))
      MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))
      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/static/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$SIM_TIME" "$MEMPOOL_SHIFT"
    done

    SHIFT=$(( SHIFT + N_POINTS * N_REP ))
  done
}

run_rwpm() {
  N_REP=40
  N_WIFI=10
  SIM_TIME=1000

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    SEED_SHIFT=$(( SHIFT + BASE_SEED ))
    MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))

    sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/rwpm/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$SIM_TIME" "$MEMPOOL_SHIFT"

    SHIFT=$(( SHIFT + N_REP ))
  done

}

run_moving() {
  N_REP=15
  VELOCITIES=(1 2)
  SIM_TIMES=("56" "27")
  INTERVALS=("1" "0.5")
  VELOCITIES_LEN=${#VELOCITIES[@]}

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < VELOCITIES_LEN; j++)); do
      VELOCITY=${VELOCITIES[$j]}
      SIM_TIME=${SIM_TIMES[$j]}
      INTERVAL=${INTERVALS[$j]}

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      SEED_SHIFT=$(( SHIFT + BASE_SEED ))
      MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))
      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/moving/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$VELOCITY" "$SIM_TIME" "$INTERVAL" "$MEMPOOL_SHIFT"
    done

    SHIFT=$(( SHIFT + VELOCITIES_LEN * N_REP ))
  done
}

echo -e "\nQueue Static (d=0) scenario"
run_static 0

echo -e "\nQueue Static (d=20) scenario"
run_static 20

echo -e "\nQueue Moving scenario"
run_moving

echo -e "\nQueue RWPM scenario"
run_rwpm
